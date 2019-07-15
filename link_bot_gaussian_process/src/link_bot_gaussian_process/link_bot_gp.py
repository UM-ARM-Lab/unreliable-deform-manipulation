import os
from time import time

import gpflow as gpf
import tensorflow as tf
import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from colorama import Fore
from matplotlib.animation import FuncAnimation

from link_bot_gaussian_process import data_reformatting
from link_bot_pycommon import experiments_util


def animate_training_data(x, arena_size=5, linewidth=7):
    T = x.shape[0]
    states = np.ndarray((T, 6))
    for t, x_t in enumerate(x):
        state = x_t[:6]
        states[t] = state

    fig = plt.figure(figsize=(10, 10))

    x_0 = states[0]
    x_0_xs = [x_0[0], x_0[2], x_0[4]]
    x_0_ys = [x_0[1], x_0[3], x_0[5]]
    line = plt.plot(x_0_xs, x_0_ys, color='black', linewidth=linewidth, zorder=1)[0]
    scat = plt.scatter(x_0_xs, x_0_ys, color=['blue', 'blue', 'green'], zorder=2, s=50)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-arena_size, arena_size])
    plt.ylim([-arena_size, arena_size])

    def update(t):
        x_t = states[t]
        x_t_xs = [x_t[0], x_t[2], x_t[4]]
        x_t_ys = [x_t[1], x_t[3], x_t[5]]
        line.set_xdata(x_t_xs)
        line.set_ydata(x_t_ys)
        offsets = np.vstack((x_t_xs, x_t_ys)).T
        scat.set_offsets(offsets)

    anim = FuncAnimation(fig, update, frames=T, interval=1)
    return anim


def predict(fwd_model, np_state, np_controls, np_duration_steps_int, steps=None, initial_variance=0.00001):
    # flatten and combine np_controls and durations
    np_controls_flat = []
    for control, duration in zip(np_controls, np_duration_steps_int):
        for i in range(duration):
            np_controls_flat.append(control)
    np_controls_flat = np.array(np_controls_flat)

    if steps is None:
        steps = np_controls_flat.shape[0]

    x_t = np.copy(np_state)

    prediction = np.zeros((steps, fwd_model.n_state))

    variances = np.ndarray((steps, fwd_model.n_state))
    for t in range(steps):
        prediction[t] = x_t
        x_t_relative = data_reformatting.make_relative_to_head(x_t)
        combined_x_t_relative = np.hstack((x_t_relative, [np_controls_flat[t]]))

        mu_delta_x_t_plus_1s, variance = fwd_model.model.predict_y(combined_x_t_relative)
        variances[t] = variance

        x_t = x_t + mu_delta_x_t_plus_1s

    return prediction, variances


def animate_predict(prediction, sdf, arena_size, linewidth=6):
    T = prediction.shape[0]

    fig = plt.figure(figsize=(10, 10))
    max = np.max(np.flipud(sdf.T))
    img = Image.fromarray(np.uint8(np.flipud(sdf.T) / max * 256))
    small_sdf = img.resize((50, 50))
    plt.imshow(small_sdf, extent=[-arena_size, arena_size, -arena_size, arena_size])

    x_0 = prediction[0]
    x_0_xs = [x_0[0], x_0[2], x_0[4]]
    x_0_ys = [x_0[1], x_0[3], x_0[5]]
    line = plt.plot(x_0_xs, x_0_ys, color='black', linewidth=linewidth, zorder=1)[0]
    scat = plt.scatter(x_0_xs, x_0_ys, color=['blue', 'blue', 'green'], linewidth=linewidth, zorder=2)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-arena_size - 1, arena_size + 1])
    plt.ylim([-arena_size - 1, arena_size + 1])

    def update(t):
        x_t = prediction[t]
        x_t_xs = [x_t[0], x_t[2], x_t[4]]
        x_t_ys = [x_t[1], x_t[3], x_t[5]]
        line.set_xdata(x_t_xs)
        line.set_ydata(x_t_ys)
        offsets = np.vstack((x_t_xs, x_t_ys)).T
        scat.set_offsets(offsets)

    anim = FuncAnimation(fig, update, frames=T, interval=100)
    return anim


class LinkBotGP:

    def __init__(self):
        """ you have to called either train or load before any of the other methods """
        self.n_data_points = None  # number of data points
        self.n_inputs = None  # input dimensionality
        self.n_outputs = None  # output dimensionality
        self.n_inducing_points = None
        self.n_state = None
        self.n_control = None
        self.maximum_training_iterations = None
        self.model_def = None
        self.model = None

    def train(self, X, Y, n_inducing_points=100, verbose=True, maximum_training_iterations=300):
        self.n_data_points, self.n_inputs = X.shape
        _, self.n_outputs = Y.shape
        self.n_inducing_points = n_inducing_points  # number of inducing points
        X = X
        Y = Y

        # noinspection SpellCheckingInspection
        self.model_def = {
            'class': gpf.kernels.SquaredExponential,
            'initial_hyper_params': {
                'lengthscales': [1.0] * self.n_inputs
            },
            'initial_likelihood_variance': [0.1] * self.n_outputs
        }
        kern_list = [self.model_def['class'](self.n_inputs, **self.model_def['initial_hyper_params']) for _ in
                     range(self.n_outputs)]
        kernel = mk.SeparateIndependentMok(kern_list)

        Zs = [X[np.random.permutation(self.n_data_points)[:self.n_inducing_points], ...].copy() for _ in
              range(self.n_outputs)]
        # initialise as list inducing features
        feature_list = [gpf.features.InducingPoints(Z) for Z in Zs]
        # create multi-output features from feature_list
        feature = mf.SeparateIndependentMof(feature_list)

        likelihood_variance = self.model_def['initial_likelihood_variance']
        likelihood = gpf.likelihoods.Gaussian(likelihood_variance)
        self.model = gpf.models.SVGP(X, Y, kernel, likelihood, feat=feature)
        opt = gpf.train.ScipyOptimizer()
        self.maximum_training_iterations = maximum_training_iterations

        t0 = time()
        opt.minimize(self.model, disp=verbose, maxiter=self.maximum_training_iterations)
        dt = time() - t0
        if verbose:
            print(Fore.YELLOW + "training time: {}s".format(dt) + Fore.RESET)

    def metadata(self):
        return {
            'n_data_points': self.n_data_points,
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'n_states': self.n_state,
            'n_control': self.n_control,
            'n_inducing_points': self.n_inducing_points,
            'maximum_training_iterations': self.maximum_training_iterations,
            'kernel_type': self.model_def['class'].__name__,
            'initial_hyper_params': self.model_def['initial_hyper_params'],
            'initial_likelihood_variance': self.model_def['initial_likelihood_variance'],
        }

    def save(self, log_path, model_name):
        saver = gpf.saver.Saver()

        full_log_path = os.path.join(os.getcwd(), 'log_data', log_path)

        experiments_util.make_log_dir(full_log_path)
        experiments_util.write_metadata(self.metadata(), model_name + '-metadata.json', log_path)
        model_path = os.path.join(full_log_path, model_name)
        print(Fore.CYAN + "Saving model to {}".format(model_path) + Fore.RESET)

        if os.path.exists(model_path):
            response = input("Do you want to overwrite {}? [y/n]".format(model_path))
            if 'y' in response:
                os.remove(model_path)
            else:
                print(Fore.YELLOW + "Answered no - aborting." + Fore.RESET)
                return

        saver.save(model_path, self.model)

    def load(self, model_path):
        print(Fore.CYAN + "Loading model from {}".format(model_path) + Fore.RESET)
        self.model = gpf.saver.Saver().load(model_path)
        self.n_data_points, self.n_inputs = self.model.X.shape
        self.n_outputs = self.model.Y.shape[1]
        self.n_inducing_points = self.model.feature.feat_list[0].Z.shape[0]
        self.n_state = self.n_outputs
        self.n_control = self.n_inputs - self.n_outputs
        # load these from the metadata file?
        self.maximum_training_iterations = None
        self.model_def = None

    @staticmethod
    def convert_triplet_action(u):
        # we should normalize the cos/sin components just because they may not be perfectly normalized
        u = np.atleast_2d(u)
        nu = np.linalg.norm(u[0, :2])
        if nu < 1e-6:
            return np.zeros((1, 2))
        return np.array([[u[0, 0] / nu * u[0, 2], u[0, 1] / nu * u[0, 2]]])

    def fwd_act(self, s, u):
        s_relative = data_reformatting.make_relative_to_head(s)
        x_star = np.hstack((s_relative, u))
        delta_mu, _ = self.model.predict_y(x_star)

        # DEBUGGING:
        # delta_mu = 0.1 * np.array([u[0, 0], u[0, 1], u[0, 0], u[0, 1], u[0, 0], u[0, 1]])

        s_next = s + delta_mu
        return s_next

    def inv_act(self, s, s_target, max_v=1.0):
        delta = s_target - s
        head_delta_mag = np.linalg.norm(delta[:, 4:6], axis=1, keepdims=True)
        x_star = np.concatenate((delta, head_delta_mag), axis=1)
        output, _ = self.model.predict_y(x_star)
        triplet_action = output[0, :3]
        pred_n_steps = output[0, 3]
        vx_vy_u = LinkBotGP.convert_triplet_action(triplet_action)

        # DEBUGGING:
        # vx_vy_u = np.atleast_2d(s_target[0, 0:2] - s[0, 0:2])
        # u_norm = np.linalg.norm(vx_vy_u)
        # if u_norm > 1:
        #     vx_vy_u = vx_vy_u / u_norm
        # pred_n_steps = np.linalg.norm(vx_vy_u) / 0.1

        return vx_vy_u, pred_n_steps
