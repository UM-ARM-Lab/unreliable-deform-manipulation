#!/usr/bin/env python

import numpy as np
import re
import scipy.optimize as optimize
from scipy.linalg import hankel


def sdf_idx_to_point(row, col, resolution, sdf_origin):
    x = (col - sdf_origin[0, 0]) * resolution[0, 0]
    y = (row - sdf_origin[1, 0]) * resolution[1, 0]
    return np.array([[y], [x]])


def point_to_sdf_idx(x, y, resolution, sdf_origin):
    row = int(x / resolution[0] + sdf_origin[0])
    col = int(y / resolution[1] + sdf_origin[1])
    return row, col


def yaw_diff(a, b):
    diff = a - b
    greater_indeces = np.argwhere(diff > np.pi)
    diff[greater_indeces] = diff[greater_indeces] - 2 * np.pi
    less_indeces = np.argwhere(diff < -np.pi)
    diff[less_indeces] = diff[less_indeces] + 2 * np.pi
    return diff


def load_sdf(filename):
    npz = np.load(filename)
    sdf = npz['sdf']
    grad = npz['sdf_gradient']
    res = npz['sdf_resolution'].reshape(2)
    origin = np.array(sdf.shape, dtype=np.int32).reshape(2) // 2
    return sdf, grad, res, origin


def state_cost(s, goal):
    return np.linalg.norm(s[0:2] - goal[0:2])


class LinearStateSpaceModelWithQuadraticCost:

    def __init__(self, N, M, L):
        """
        N: dimensionality of the full state
        M: dimensionality in the reduced state
        L: dimensionality in the actions
        """
        self.N = N
        self.M = M
        self.L = L
        self.A = np.ndarray((M, N))
        self.B = np.ndarray((1, M))
        self.C = np.ndarray((M, L))
        self.D = np.ndarray((M, M))

    def size(self):
        return self.A.size + self.B.size + self.C.size + self.D.size

    def from_matrices(self, A, B, C, D):
        assert A.size == self.A.size
        assert B.size == self.B.size
        assert C.size == self.C.size
        assert D.size == self.D.size

        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def from_params(self, params):
        assert len(params) == self.size(), "Expected {:d} params, fot {:d}".format(self.size(), len(params))
        params = np.array(params)
        n_A = self.A.size
        self.A = params[0:n_A].reshape(self.A.shape)
        n_B = n_A + self.B.size
        self.B = params[n_A:n_B].reshape(self.B.shape)
        n_C = n_B + self.C.size
        self.C = params[n_B:n_C].reshape(self.C.shape)
        n_D = n_C + self.D.size
        self.D = params[n_C:n_D].reshape(self.D.shape)

    def to_params(self):
        return np.concatenate((self.A.flatten(),
                               self.B.flatten(),
                               self.C.flatten(),
                               self.D.flatten()))

    def reduce(self, s):
        return np.dot(self.A, s)

    def predict_from_s(self, s, u, dt):
        o = np.dot(self.A, s)
        o_ = o + (np.dot(self.B, o) + np.dot(self.C, u)) * dt
        return o_

    def predict_from_o(self, o, u, dt):
        o_ = o + (np.dot(self.B, o) + np.dot(self.C, u)) * dt
        return o_

    def cost_of_s(self, s, g):
        o = np.dot(self.A, s)
        o_g = np.dot(self.A, g)
        return np.dot((o_g - o).T, np.dot(self.D, o_g - o))

    def cost_of_o(self, o, g):
        o_g = np.dot(self.A, g)
        return np.dot((o_g - o).T, np.dot(self.D, o_g - o))

    def predict_cost_of_s(self, s, u, dt, g):
        return self.cost_of_o(self.predict_from_s(s, u, dt), g)

    def predict_cost(self, o, u, dt, g):
        return self.cost_of_o(self.predict_from_o(o, u, dt), g)

    def __repr__(self):
        return "Model reduction Matrix: {}\n Dynamics matrices: {}, {}\n Cost Matrix: {}".format(self.A, self.B, self.C,
                                                                                                 self.D)

    def save(self, outfile):
        np.savez(outfile, A=self.A, B=self.B, C=self.C, D=self.D)

    def load(self, infile):
        matrices = np.aoad(infile)
        self.A = matrices['A']
        self.B = matrices['B']
        self.C = matrices['C']
        self.D = matrices['D']


def train(data, model, goal, dt, objective_func, initial_params=None, tol=None, method='Powell', options={}, **kwargs):
    """
    mutates the model that was passed in
    """

    def __objective(params):
        model.from_params(params)
        return objective_func(model=model, g=goal, data=data, dt=dt, **kwargs)

    if initial_params is None:
        initial_params = np.random.randn(model.size())

    result = optimize.minimize(__objective, initial_params, method=method, options=options)

    if not result.success:
        print("Status: {:d}, Message: {:s}".format(result.status, result.message))
        return result
    print('Finished in {:d} iterations'.format(result.nit))


def current_cost(model, g, data):
    err = np.zeros(len(data))
    for i, (s, u, s_, c, c_) in enumerate(data):
        err[i] = model.cost_of_s(s, g) - c

    return (err ** 2).mean()


def state_prediction(model, g, data, dt):
    err = np.zeros(len(data))
    for i, (s, u, s_, c, c_) in enumerate(data):
        err[i] = np.linalg.norm(model.predict_from_s(s, u, dt) - model.reduce(s_))

    return (err ** 2).mean()


def cost_prediction(model, g, data, dt):
    err = np.zeros(len(data))
    for i, (s, u, s_, c, c_) in enumerate(data):
        o = model.reduce(s)
        err[i] = np.linalg.norm(model.predict_cost(o, u, dt, g) - c_)

    return (err ** 2).mean()


def state_prediction_objective(model, g, data, dt, alpha=0.5, regularization=1e-4):
    """ return: MSE over all training examples """
    obj = alpha * state_prediction(model, g, data, dt)
    obj += (1 - alpha) * current_cost(model, g, data)
    obj += regularization * np.linalg.norm(model.to_params())
    return obj


def cost_prediction_objective(model, g, data, dt, alpha=0.5, regularization=1e-4):
    """ return: MSE over all training examples """
    obj = alpha * cost_prediction(model, g, data, dt)
    obj += (1 - alpha) * current_cost(model, g, data)
    obj += regularization * np.linalg.norm(model.to_params())
    return obj


def load_data(log_file, g, extract_func):
    log_data = np.loadtxt(log_file)
    new_data = []
    for i in range(log_data.shape[0] - 1):
        s, u, c, = extract_func(log_data[i], g)
        s_, u_, c_, = extract_func(log_data[i + 1], g)
        new_datum = [s, u, s_, c, c_]
        new_data.append(new_datum)
    return new_data


def one_link_pos_extractor(row, g):
    s = np.expand_dims(row[0:4], axis=1)
    u = np.expand_dims(row[4:6], axis=1)
    c = (row[0] - g[0]) ** 2 + (row[1] - g[1]) ** 2
    return s, u, c


def two_link_pos_extractor(row, g):
    s = np.expand_dims(row[0:6], axis=1)
    u = np.expand_dims(row[8:10], axis=1)
    c = (row[0] - g[0]) ** 2 + (row[1] - g[1]) ** 2
    return s, u, c


def two_link_pos_vel_extractor(row, g):
    # 0   1   2   3   4   5   6   7   8   9  10  11
    # x0 y0 vx0 vy0  x1  y1 vx1 vy1  x2  y2 vx2 vy2
    s = np.expand_dims(row[[0, 1, 4, 5, 8, 9]], axis=1)
    u = np.expand_dims(row[[10, 11]], axis=1)
    c = (row[0] - g[0]) ** 2 + (row[1] - g[1]) ** 2
    return s, u, c


def five_link_pos_vel_extractor(row, g):
    # 0   1   2   3   4   5   6   7   8   9  10  11 ...
    # x0 y0 vx0 vy0  x1  y1 vx1 vy1  x2  y2 vx2 vy2 ...
    s = np.expand_dims(row[[0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21]], axis=1)
    u = np.expand_dims(row[[22, 23]], axis=1)
    c = (row[0] - g[0]) ** 2 + (row[1] - g[1]) ** 2
    return s, u, c


def link_pos_vel_extractor(N):
    s_indeces = []
    for i in range(0, 2 * N, 4):
        s_indeces.extend([i, i + 1])

    def __link_pos_vel_extractor(row, g):
        s = np.expand_dims(row[s_indeces], axis=1)
        u = np.expand_dims(row[[2 * N - 2, 2 * N - 1]], axis=1)
        c = (row[0] - g[0]) ** 2 + (row[1] - g[1]) ** 2
        return s, u, c

    return __link_pos_vel_extractor


def link_pos_vel_extractor2(N):
    s_indeces = []
    for i in range(0, 2 * N, 4):
        s_indeces.extend([i, i + 1])

    def __link_pos_vel_extractor(row):
        s = np.expand_dims(row[s_indeces], axis=1)
        u = np.expand_dims(row[[2 * N - 2, 2 * N - 1]], axis=1)
        return s, u

    return __link_pos_vel_extractor


def link_pos_vel_extractor2_indeces():
    return [0, 1, 4, 5, 8, 9, 10, 11]


def fake_indeces():
    return [0, 1, 2, 3]


def load_train_test(filename, N, M, L, g, extract_func):
    log_data = np.loadtxt(filename)
    n_training_samples = log_data.shape[0]
    train_x = np.ndarray((n_training_samples - 1, 3 * N + L))
    train_y = np.ndarray((n_training_samples - 1, 2))
    for i in range(n_training_samples - 1):
        s, u, c, = extract_func(log_data[i], g)
        s_, u_, c_, = extract_func(log_data[i + 1], g)
        train_x[i] = np.concatenate((s.flatten(), s_.flatten(), g.flatten(), u.flatten()))
        train_y[i][0] = c
        train_y[i][1] = c_

    return n_training_samples, train_x, train_y


def parse_dataset_name(dataset, log_data):
    matches = re.search(r"(\d+)_(\d+)_.*", dataset)
    if not matches:
        print("could not parse dataset name")
        return
    try:
        trajectory_length_during_collection = int(matches.group(2))

    except ValueError:
        print("could not convert regex matches in filename to integers")
        return

    return trajectory_length_during_collection


def subsequences(v, q):
    m = v.shape[0] - q + 1
    return hankel(v[:m], v[m - 1:])


def load_train(filename, N, L, extract_func, n_steps=1):
    log_data = np.loadtxt(filename)
    n_training_samples = log_data.shape[0]
    n_trajs = int(n_training_samples / (n_steps + 1))
    train_x = np.ndarray((n_steps + 1, N + L, n_trajs))
    for k, d in enumerate(log_data):
        s, u = extract_func(d)
        i = int(k / (n_steps + 1))
        j = k % (n_steps + 1)
        train_x[j, :, i] = np.concatenate((s.flatten(), u.flatten()))


def plot_gz_data(plt, new_data):
    plt.figure()
    plt.title(r"Full State ($s$)")
    plt.plot([s[0][0, 0] for s in new_data], label='x1')
    plt.plot([s[0][1, 0] for s in new_data], label='y1')
    plt.plot([s[0][2, 0] for s in new_data], label='x2')
    plt.plot([s[0][3, 0] for s in new_data], label='y2')
    plt.ylabel("meters")
    plt.xlabel("time (steps)")
    plt.legend()

    plt.figure()
    plt.title(r"Control Input ($u$)")
    plt.plot([s[1][0, 0] for s in new_data], label='vx')
    plt.plot([s[1][1, 0] for s in new_data], label='vy')
    plt.ylabel("m/s")
    plt.xlabel("time (steps)")
    plt.legend()

    plt.figure()
    plt.title(r"Cost ($c$)")
    plt.plot([s[4] for s in new_data])
    plt.xlabel("time (steps)")

    plt.show()


def plot_gz_data_v2(plt, new_data):
    plt.figure()
    plt.title(r"Full State ($s$)")
    plt.plot([s[0][0, 0] for s in new_data], label='x1')
    plt.plot([s[0][1, 0] for s in new_data], label='y1')
    plt.plot([s[0][2, 0] for s in new_data], label='x2')
    plt.plot([s[0][3, 0] for s in new_data], label='y2')
    plt.plot([s[0][4, 0] for s in new_data], label='x3')
    plt.plot([s[0][5, 0] for s in new_data], label='y3')
    plt.ylabel("meters")
    plt.xlabel("time (steps)")
    plt.legend()

    plt.figure()
    plt.title(r"Position of first point")
    plt.scatter([s[0][0, 0] for s in new_data], [s[0][1, 0] for s in new_data])
    plt.ylabel("x (m)")
    plt.xlabel("y (m)")
    plt.legend()

    plt.figure()
    plt.title(r"Control Input ($u$)")
    plt.plot([s[1][0, 0] for s in new_data], label='fx')
    plt.plot([s[1][1, 0] for s in new_data], label='fy')
    plt.ylabel("m/s")
    plt.xlabel("time (steps)")
    plt.legend()

    plt.figure()
    plt.title(r"Cost ($c$)")
    plt.plot([s[4] for s in new_data])
    plt.xlabel("time (steps)")

    plt.show()


def train_and_eval(model, data, g, dt, objective, initial_params=None, alpha=0.5, regularization=1e-5,
                   print_model=True):
    train(data, model, g, dt, objective, initial_params)
    return eval_model(model, data, g, dt, alpha, regularization, print_model)


def eval_model(model, data, g, dt, alpha=0.5, regularization=1e-5, print_model=True):
    pred_state_loss = state_prediction(model, g, data, dt)
    cost_loss = current_cost(model, g, data)
    pred_cost_loss = cost_prediction(model, g, data, dt)
    pred_state_curr_cost_loss = state_prediction_objective(model, g, data, dt, alpha=alpha,
                                                           regularization=regularization)
    pred_cost_curr_cost_loss = cost_prediction_objective(model, g, data, dt, alpha=alpha, regularization=regularization)
    reg_loss = regularization * np.linalg.norm(model.to_params())
    eval_str = "Loss Components:\n"
    eval_str += "\tcurrent cost: {}\n".format(cost_loss)
    eval_str += "\tpredict next latent state: {}\n".format(pred_state_loss)
    eval_str += "\tpredict next cost: {}\n".format(pred_cost_loss)
    eval_str += "\tregularization: {}\n".format(reg_loss)
    eval_str += "Complete Losses:\n"
    eval_str += "\tpredict next latent state and current cost: {}\n".format(pred_state_curr_cost_loss)
    eval_str += "\tpredict next cost and current cost: {}\n".format(pred_cost_curr_cost_loss)
    print(eval_str)
    if print_model:
        print(model)
    return cost_loss, pred_state_loss, pred_cost_loss, reg_loss


def mean_dot_product(model, data, dt, g):
    costs = [d[3:5] for d in data]
    sum_of_dots = 0
    for i, d in enumerate(data):
        s = d[0]
        u = d[1]
        c_hat = model.cost_of_s(s, g)[0, 0]
        c_hat_ = model.predict_cost_of_s(s, u, dt, g)[0, 0]
        sum_of_dots += np.dot(c_hat_ - c_hat, costs[i][1] - costs[i][0])[0]
    return sum_of_dots / len(data)


def plot_x_rollout(plt, model, data, dt, s0, g):
    actions = [d[1] for d in data]
    o = model.reduce(s0)
    predicted_total_cost = 0.0
    o_s = [o]
    for u in actions:
        c_hat = model.cost_of_o(o, g)
        o = model.predict_from_o(o, u, dt)
        o_s.append(o)
        predicted_total_cost += c_hat

    states = [d[0] for d in data]
    plt.figure()
    plt.plot([s[0, 0] for s in states], label='true x1')
    plt.plot(np.squeeze(o_s), label='latent space o', linewidth=3, linestyle='--')
    plt.xlabel("time steps")
    plt.ylabel("o")
    plt.legend()
    plt.show()

    return predicted_total_cost


def plot_xy_rollout(plt, model, data, dt, s0, g):
    actions = [d[1] for d in data]
    o = model.reduce(s0)
    predicted_total_cost = 0.0
    o_s = [o]
    for u in actions:
        c_hat = model.cost_of_o(o, g)
        o = model.predict_from_o(o, u, dt)
        o_s.append(o)
        predicted_total_cost += c_hat

    states = [d[0] for d in data]
    plt.figure()
    plt.plot([s[0, 0] for s in states], label='true x1')
    plt.plot([s[1, 0] for s in states], label='true y1')
    plt.plot(np.squeeze(o_s), label='latent space o', linewidth=3, linestyle='--')
    plt.xlabel("time steps")
    plt.ylabel("o")
    plt.legend()
    plt.show()

    return predicted_total_cost


def plot_cost(plt, model, data, dt, g):
    plt.figure()

    costs = [d[3] for d in data]
    plt.scatter(np.arange(len(data)), costs, label='true cost')

    for i, d in enumerate(data):
        s = d[0]
        u = d[1]
        c_hat = model.cost_of_s(s, g)[0, 0]
        c_hat_ = model.predict_cost_of_s(s, u, dt, g)[0, 0]
        plt.plot([i, i + 1], [c_hat, c_hat_], color='red')
        plt.scatter([i], [c_hat], color='green', s=10)

    plt.title("Estimated vs True Cost")
    plt.xlabel("time steps")
    plt.ylabel("o")
    plt.legend()
    plt.show()


def plot_o_rollout(plt, model, data, dt, g):
    states = [d[0] for d in data]
    actions = [d[1] for d in data]
    o = model.reduce(states[0])
    predicted_total_cost = 0.0
    o_rollout = [o]
    o_s = []
    for s, u in zip(states, actions):
        o_s.append(model.reduce(s))
        c_hat = model.cost_of_o(o, g)
        o = model.predict_from_o(o, u, dt)
        o_rollout.append(o)
        predicted_total_cost += c_hat

    plt.figure()
    plt.title("Rollout in Reduction Space")
    plt.plot(np.squeeze(o_s), label='reduction of s_t')
    plt.plot(np.squeeze(o_rollout), label='rollout from initial s_t', linewidth=3, linestyle='--')
    plt.xlabel("time steps")
    plt.ylabel("o")
    plt.legend()
    plt.show()


def plot_costmap(plt, model, data, g, resolution=0.1, samples=5, spread=2):
    colors = {}
    min_sample = None
    min_sample_cost = 1e9
    for d in data:
        state = d[0]
        for i in range(samples):
            s = state + np.random.randn(*state.shape) * spread
            c = model.cost_of_s(s, g)[0, 0]
            xy = (s[0, 0], s[1, 0])
            colors[xy] = c
            if c < min_sample_cost:
                min_sample_cost = c
                min_sample = s

    plt.figure(figsize=(10, 10))
    xs = [k[0] for k in colors.keys()]
    ys = [k[1] for k in colors.keys()]
    plt.scatter(xs, ys, c=colors.values(), s=10, cmap='pink')
    plt.axis("equal")
    return min_sample, min_sample_cost


def plot_costmap_2(plt, model, data, g, resolution=0.1, minimum=-5, maximum=5):
    N = int((maximum - minimum) / resolution)
    colors = np.ndarray((N, N))
    for i in range(N):
        x = minimum + resolution * i
        for j in range(N):
            y = minimum + resolution * j
            s = np.array([[x], [y], [0], [0], [0], [0]])
            c = model.cost_of_s(s, g)[0, 0]
            colors[N - j - 1, i] = c

    plt.imshow(colors, interpolation=None, extent=[minimum, maximum, minimum, maximum])


def policy_quiver(model, action_selector, goal, ax, cx, cy, r, m, scale=10):
    x = []
    y = []
    u = []
    v = []
    for s1 in np.arange(cx - m, cx + m + r, r):
        for s2 in np.arange(cy - m, cy + m + r, r):
            o = model.reduce(np.array([[s1], [s2], [0], [0], [0], [0]]))
            a = action_selector.act(o)
            x.append(s1)
            y.append(s2)
            u.append(a[0, 0, 0])
            v.append(a[0, 1, 0])
    q = ax.quiver(x, y, u, v, scale=scale, width=scale / 20000.0)


def random_goals(n_goals):
    goals = []
    for _ in range(n_goals):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        theta1 = np.random.uniform(-np.pi / 2, np.pi / 2)
        theta2 = np.random.uniform(-np.pi / 2, np.pi / 2)
        x1 = x + np.cos(theta1)
        y1 = y + np.sin(theta1)
        x2 = x1 + np.cos(theta2)
        y2 = y1 + np.sin(theta2)
        g = np.array([[x], [y], [x1], [y1], [x2], [y2]])
        goals.append(g)
    return goals
