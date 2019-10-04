import json
import os
import pathlib
from time import time
from typing import Optional, List

import gpflow as gpf
import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from colorama import Fore
from matplotlib.animation import FuncAnimation

from link_bot_gaussian_process import data_reformatting
from link_bot_gaussian_process.my_svgp import MySVGP
from link_bot_pycommon import experiments_util


def predict_one(fwd_model, np_state, np_controls, steps=None, initial_variance=0.00001):
    if steps is None:
        steps = np_controls.shape[0] + 1

    x_t = np.copy(np_state)

    prediction = np.zeros((steps, np_state.shape[1]))
    prediction[0] = np_state[0]

    variances = np.zeros((steps, np_state.shape[1]))
    for t in range(steps - 1):
        prediction[t + 1] = x_t
        x_t_relative = data_reformatting.make_relative_to_head(x_t)[:, :-2]
        combined_x_t_relative = np.hstack((x_t_relative, [np_controls[t]]))

        mu_delta_x_t_plus_1s, variance = fwd_model.model.predict_y(combined_x_t_relative)
        variances[t + 1] = variance

        x_t = x_t + mu_delta_x_t_plus_1s

    return prediction, variances


def animate_predict(prediction: np.ndarray,
                    y_rope_configurations: np.ndarray,
                    extent: List[float],
                    sdf: Optional[np.ndarray] = None,
                    linewidth: float = 6,
                    example_idx: int = None):
    """
    :param prediction: Tx6 array
    :param y_rope_configurations: Tx6 array
    :param extent: [xmin, xmax, ymin, ymax]
    :param sdf:
    :param linewidth:
    :param example_idx:
    :return:
    """
    T = prediction.shape[0]

    fig = plt.figure()
    if sdf is not None:
        plt.imshow(np.flipud(sdf.T) > 0, extent=extent)

    pred_x_0 = prediction[0]
    pred_x_0_xs = [pred_x_0[0], pred_x_0[2], pred_x_0[4]]
    pred_x_0_ys = [pred_x_0[1], pred_x_0[3], pred_x_0[5]]
    pred_line = plt.plot(pred_x_0_xs, pred_x_0_ys, color='black', linewidth=linewidth, zorder=1)[0]
    pred_scatt = plt.scatter(pred_x_0_xs, pred_x_0_ys, color=['blue', 'blue', 'green'], linewidth=linewidth, zorder=2)

    true_x_0 = y_rope_configurations[0]
    true_x_0_xs = [true_x_0[0], true_x_0[2], true_x_0[4]]
    true_x_0_ys = [true_x_0[1], true_x_0[3], true_x_0[5]]
    true_line = plt.plot(true_x_0_xs, true_x_0_ys, color='red', linewidth=linewidth, zorder=1)[0]
    true_scatt = plt.scatter(true_x_0_xs, true_x_0_ys, color=['red', 'red', 'orange'], linewidth=linewidth, zorder=2)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.xlim(extent[0:2])
    plt.ylim(extent[2:4])
    if example_idx:
        plt.title(example_idx)
    time_text_handle = plt.text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5},
                                bbox=dict(facecolor='black', alpha=0.5))

    def update(t):
        pred_x_t = prediction[t]
        pred_x_t_xs = [pred_x_t[0], pred_x_t[2], pred_x_t[4]]
        pred_x_t_ys = [pred_x_t[1], pred_x_t[3], pred_x_t[5]]
        pred_line.set_xdata(pred_x_t_xs)
        pred_line.set_ydata(pred_x_t_ys)
        pred_offsets = np.vstack((pred_x_t_xs, pred_x_t_ys)).T
        pred_scatt.set_offsets(pred_offsets)

        true_x_t = y_rope_configurations[t]
        true_x_t_xs = [true_x_t[0], true_x_t[2], true_x_t[4]]
        true_x_t_ys = [true_x_t[1], true_x_t[3], true_x_t[5]]
        true_line.set_xdata(true_x_t_xs)
        true_line.set_ydata(true_x_t_ys)
        true_offsets = np.vstack((true_x_t_xs, true_x_t_ys)).T
        true_scatt.set_offsets(true_offsets)

        time_text_handle.set_text("t={}".format(t))

    anim = FuncAnimation(fig, update, frames=T, interval=250)
    return anim


class LinkBotGP:

    def __init__(self, rng_type=None):
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
        self.rng = None
        self.min_steps = 1
        self.max_steps = 1
        self.rng_type = rng_type
        self.dataset_hparams = None
        self.metadata = None
        self.dt = None

    def initialize_rng(self, min_steps, max_steps):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.rng = self.rng_type()

    def train(self, X, Y, beta, dataset_hparams, n_inducing_points=100, verbose=True, maximum_training_iterations=300):
        self.n_data_points, self.n_inputs = X.shape
        _, self.n_outputs = Y.shape
        self.n_state = 6
        self.n_control = 2
        self.n_inducing_points = n_inducing_points  # number of inducing points
        self.dataset_hparams = dataset_hparams
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
        self.model = MySVGP(X, Y, kernel, likelihood, feat=feature, beta=beta)

        self.maximum_training_iterations = maximum_training_iterations

        # Now that all the variables related to training have been set, we can build our metadata
        self.build_metadata()

        optimizer = gpf.train.AdamOptimizer()
        optimizer_tensor = optimizer.make_optimize_tensor(self.model)
        session = gpf.get_default_session()

        t0 = time()
        for _ in range(self.maximum_training_iterations):
            session.run(optimizer_tensor)
        training_time = time() - t0

        if verbose:
            print(self.model.compute_log_likelihood())
            print(Fore.YELLOW + "training time: {:7.3f}s".format(training_time) + Fore.RESET)

    def get_metadata(self):
        return self.metadata

    def build_metadata(self):
        self.metadata = {
            'n_data_points': self.n_data_points,
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'n_state': self.n_state,
            'n_control': self.n_control,
            'n_inducing_points': self.n_inducing_points,
            'maximum_training_iterations': self.maximum_training_iterations,
            'kernel_type': self.model_def['class'].__name__,
            'initial_hyper_params': self.model_def['initial_hyper_params'],
            'initial_likelihood_variance': self.model_def['initial_likelihood_variance'],
            'dataset_hparams': self.dataset_hparams,
        }

    def save(self, log_path, model_name):
        saver = gpf.saver.Saver()

        full_log_path = pathlib.Path.cwd() / 'log_data' / log_path

        experiments_util.make_log_dir(full_log_path)
        experiments_util.write_metadata(self.get_metadata(), model_name + '-metadata.json', log_path)
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

    def load(self, model_path: pathlib.Path):
        print(Fore.CYAN + "Loading model from {}".format(model_path) + Fore.RESET)
        self.model = gpf.saver.Saver().load(model_path)

        metadata = json.load(open((str(model_path) + '-metadata.json'), 'r'))
        self.metadata = metadata
        self.dataset_hparams = metadata['dataset_hparams']
        self.n_data_points = metadata['n_data_points']
        self.n_inputs = metadata['n_inputs']
        self.n_outputs = metadata['n_outputs']
        self.n_inducing_points = metadata['n_inducing_points']
        self.n_state = metadata['n_state']
        self.n_control = metadata['n_control']
        self.dt = metadata['dataset_hparams']['dt']
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
        s_relative = s_relative[:, :-2]
        x_star = np.hstack((s_relative, u))
        delta_mu, _ = self.model.predict_y(x_star)

        s_next = s + delta_mu
        return s_next

    def inv_act(self, s, s_target, max_v=1.0):
        delta = s_target - s
        head_delta_mag = np.linalg.norm(delta[:, 4:6], axis=1, keepdims=True)
        x_star = np.concatenate((delta, head_delta_mag), axis=1)
        u, _ = self.model.predict_y(x_star)
        # normalize in case the GP outputs sin/cos that are slightly > 1
        u_norm = np.linalg.norm(u)
        if u_norm > 1:
            u = u / u_norm

        return u

    def dumb_inv_act(self, s, s_target, max_v=0.15):
        min_distance = np.inf
        min_u = None
        for i in range(10):
            theta = np.random.uniform(-np.pi, np.pi)
            u = np.array([[max_v * np.cos(theta), max_v * np.sin(theta)]])
            s_next = self.fwd_act(s, u)
            distance = np.linalg.norm(s_next - s_target)
            if distance < min_distance:
                min_distance = distance
                min_u = u

        return min_u

    def predict(self, first_states, batch_actions):
        """
        It's T+1 because it includes the first state
        :param np_first_states: [batch, 6]
        :param np_actions: [batch, T, 2]
        :return: [batch, T+1, 3, 2]
        """
        predictions = []
        for first_state, actions in zip(first_states, batch_actions):
            np_state = np.expand_dims(first_state, axis=0)
            prediction, _ = predict_one(self, np_state, actions)
            predicted_points = prediction.reshape([-1, 3, 2])
            predictions.append(predicted_points)
        return predictions
