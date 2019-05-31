import os
from time import time

import gpflow as gpf
import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk
from link_bot_gaussian_process import data_reformatting
import numpy as np
from colorama import Fore

from link_bot_notebooks import experiments_util


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

    def train(self, X, Y, M=100, verbose=True, maximum_training_iterations=100):
        self.n_data_points, self.n_inputs = X.shape
        _, self.n_outputs = Y.shape
        self.n_inducing_points = M  # number of inducing points
        X = X
        Y = Y

        # noinspection SpellCheckingInspection
        self.model_def = {
            'class': gpf.kernels.SquaredExponential,
            'initial_hyper_params': {
                'lengthscales': [1.0]*self.n_inputs
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
            'n_states': self.n_outputs,
            'n_control': self.n_inputs - self.n_outputs,
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
    def convert_u(u):
        return np.array([[u[0, 0] * u[0, 2], u[0, 1] * u[0, 2]]])

    def fwd_act(self, s, u):
        s = s.T
        s_relative = data_reformatting.make_relative_to_head(s)
        x_star = np.hstack((s_relative, u))
        mu, var = self.model.predict_y(x_star)
        mu = s + mu
        return mu

    def inv_act(self, s, s_target, max_v=1.0):
        x_star = (s_target - s).T
        mu, var = self.model.predict_y(x_star)

        u = np.array([[mu[0, 0] * mu[0, 2], mu[0, 1] * mu[0, 2]]])

        # TODO: is this a bad thing?
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-9:
            if u_norm > max_v:
                scaling = max_v
            else:
                scaling = u_norm
            mu = mu * scaling / u_norm

        return mu
