import argparse

import matplotlib.pyplot as plt
import bisect
import numpy as np
from scipy.optimize import root
from scipy.spatial import cKDTree

from link_bot_notebooks import toy_problem_optimization_common as tpoc


def sdf_to_dict(sdf):
    sdf_dict = {}
    for row, col in np.ndindex(sdf.shape):
        d = sdf[row, col]
        if d not in sdf_dict:
            sdf_dict[d] = []
        sdf_dict[d].append((row, col))
    for k, v in sdf_dict.items():
        sdf_dict[k] = np.array(v)
    return sdf_dict


def Rk(params):
    a1 = params[0]
    a2 = params[1]
    a3 = params[2]
    return np.array([
        [a1, 0],
        [0, a1],
        [a2, 0],
        [0, a2],
        [a3, 0],
        [0, a3],
    ])


def select_data_at_constraint_boundary(states, constraints):
    data_at_boundary = []
    T = states.shape[1]
    for state_traj, constraint_traj in zip(states, constraints):
        for t_idx in range(T - 1):
            current_constraint = constraint_traj[t_idx]
            next_constraint = constraint_traj[t_idx + 1]
            # xor operator
            if bool(current_constraint) ^ bool(next_constraint):
                average_state = (state_traj[t_idx] + state_traj[t_idx + 1]) / 2
                data_at_boundary.append(average_state)
    data_at_boundary = np.array(data_at_boundary)
    return data_at_boundary


def nn_correspondences(source, target):
    """
    source is a N_d by 2 matrix of points in R^2, which are the result of transforming data_at_constraint_boundary by R_k
    source is CONSTANT
    target is a N_m by 2 matrix of points in R^2, which is sdf_points_at_constraint_boundary
    we need to find the nearest point in target to each point in source
    """
    indeces = np.ndarray([source.shape[0]], dtype=np.int)
    for i, source_point in enumerate(source):
        min_distance = np.inf
        min_idx = None
        for idx, target_point in enumerate(target):
            dist = np.linalg.norm(target_point - source_point)
            if dist < min_distance:
                min_distance = dist
                min_idx = idx
        indeces[i] = min_idx

    return indeces


def nearby_sdf_lookup(sdf_dict, origin, res, threshold):
    sorted_distance_keys = sdf_dict['sorted_distance_keys']
    nearest_distance_key = bisect.bisect_left(sorted_distance_keys, threshold)
    left_idx = 0
    right_idx = 0
    left_idx_near = True
    right_idx_near = True
    nearby_distance_keys = [nearest_distance_key]
    while left_idx_near or right_idx_near:
        if left_idx_near:
            left_idx += 1
            left_d = sorted_distance_keys[nearest_distance_key - left_idx]
            if abs(left_d - threshold) > 0.005:
                left_idx_near = False
            else:
                nearby_distance_keys.append(nearest_distance_key - left_idx)
        if right_idx_near:
            right_idx += 1
            right_d = sorted_distance_keys[nearest_distance_key + right_idx]
            if abs(right_d - threshold) > 0.005:
                right_idx_near = False
            else:
                nearby_distance_keys.append(nearest_distance_key + right_idx)

    indeces = np.concatenate([sdf_dict[sorted_distance_keys[k]] for k in nearby_distance_keys])
    points = tpoc.sdf_indeces_to_point(indeces, resolution=res, origin=origin)
    return points


correspondence_cache = {}


def func(sdf_dict, origin, res, data_at_constraint_boundary, params):
    global correspondence_cache

    R_k = Rk(params)

    # iterate over the data and find the data points which are on the boundary of collision, and take their average
    transformed_data = data_at_constraint_boundary @ R_k

    # sdf_threshold = params[-1]
    sdf_threshold = 0.2
    if sdf_threshold in correspondence_cache:
        sdf_points_at_threshold, kd_tree = correspondence_cache[sdf_threshold]
        _, correspondence_guess = kd_tree.query(transformed_data)
    else:
        sdf_points_at_threshold = nearby_sdf_lookup(sdf_dict, origin, res, sdf_threshold)
        kd_tree = cKDTree(data=sdf_points_at_threshold)
        correspondence_cache[sdf_threshold] = (sdf_points_at_threshold, kd_tree)
        _, correspondence_guess = kd_tree.query(transformed_data)

    # correspondence_guess = nn_correspondences(transformed_data, sdf_points_at_threshold)
    corresponding_sdf_points = sdf_points_at_threshold[correspondence_guess]

    error = np.linalg.norm(corresponding_sdf_points - transformed_data, axis=1)
    # loss = np.mean(error)
    # return loss, corresponding_sdf_points
    return error, corresponding_sdf_points


def attempt_minimize(sdf_dict, origin, res, data_at_constraint_boundary, out_params=[]):
    def _func(params):
        print(params)
        loss, corresponding_sdf_points = func(sdf_dict, origin, res, data_at_constraint_boundary, params)
        out_params.append(corresponding_sdf_points)
        return loss

    # initial_a = np.random.randn(3)
    initial_a = [1, 0, 0]
    # initial_threshold = np.random.uniform(0.01, 0.40, size=1)
    # initial_threshold = [0.2]
    # initial_params = np.concatenate((initial_a, initial_threshold))
    # print(initial_params)
    # bounds = Bounds([-np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, 1])
    sol = root(_func, x0=initial_a, jac=None, method='lm')
    return sol


def main():
    np.set_printoptions(suppress=True, linewidth=200, precision=2)

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('sdf')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    plt.style.use('slides')

    data = np.load(args.data)

    sdf, _, sdf_resolution, sdf_origin = tpoc.load_sdf(args.sdf)

    states = data['states']
    constraints = data['constraints']

    # convert the sdf into a dict of {distance: [(row, col), (row, col), ...], distance:, [...], ...}
    sdf_dict = sdf_to_dict(sdf)
    # FIXME: sneaky add of this key to keep the data together
    sorted_distance_keys = sorted(sdf_dict.keys())
    sdf_dict['sorted_distance_keys'] = sorted_distance_keys

    data_at_constraint_boundary = select_data_at_constraint_boundary(states, constraints)

    success_threshold = 0.02
    failures = 0
    maximum_iterations = 100
    for attempt in range(maximum_iterations):
        out_params = []
        sol = attempt_minimize(sdf_dict, sdf_origin, sdf_resolution, data_at_constraint_boundary, out_params)
        mean_error = np.mean(sol.fun)
        print(mean_error, sol.x)
        sdf_points_at_threshold = out_params[0]
        if mean_error < success_threshold:
            print("mean error: {:0.4f}".format(mean_error))
            print("parameters: {}".format(sol.x))
            break
    # http://book.pythontips.com/en/latest/for_-_else.html
    else:
        failures += 1
    print("# failures: {}".format(failures))

    if args.plot:
        # show all location of the tail overlayed on the SDF
        plt.figure()
        subsample = 10
        plt.imshow(np.flipud(sdf.T), extent=[-5, 5, -5, 5])
        candidate_xs = sdf_points_at_threshold[::subsample, 0]
        candidate_ys = sdf_points_at_threshold[::subsample, 1]
        plt.scatter(candidate_xs, candidate_ys, c='y', s=1, alpha=0.8, label='points in sdf at constraint boundary')
        plt.scatter(data_at_constraint_boundary[:, 4], data_at_constraint_boundary[:, 5], c='r', s=10,
                    label='head at constraint boundary')
        plt.legend()
        plt.axis("equal")
        plt.show()


if __name__ == '__main__':
    main()
