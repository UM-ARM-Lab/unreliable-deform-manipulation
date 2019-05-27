import os
from time import time

from colorama import Fore
import gpflow as gpf
import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk
import numpy as np

from latent_constraint_learning import experiments_util


class LinkBotGP:

    def __init__(self, X, Y, M=100):
        self.N, self.D = X.shape
        _, self.P = Y.shape
        self.M = M  # number of inducing points
        self.X = X
        self.Y = Y

        # noinspection SpellCheckingInspection
        self.model_def = {
            'class': gpf.kernels.SquaredExponential,
            'initial_hyper_params': {
                'lengthscales': 1.0
            },
            'initial_likelihood_variance': [0.1] * self.P
        }
        self.kern_list = [self.model_def['class'](self.D, **self.model_def['initial_hyper_params']) for _ in
                          range(self.P)]
        self.kernel = mk.SeparateIndependentMok(self.kern_list)

        self.Zs = [X[np.random.permutation(self.N)[:self.M], ...].copy() for _ in range(self.P)]
        # initialise as list inducing features
        self.feature_list = [gpf.features.InducingPoints(Z) for Z in self.Zs]
        # create multi-output features from feature_list
        self.feature = mf.SeparateIndependentMof(self.feature_list)

        self.likelihood_variance = self.model_def['initial_likelihood_variance']
        self.likelihood = gpf.likelihoods.Gaussian(self.likelihood_variance)
        self.model = gpf.models.SVGP(self.X, self.Y, self.kernel, self.likelihood, feat=self.feature)
        self.opt = gpf.train.ScipyOptimizer()
        self.maximum_training_iterations = None

    def train(self, verbose=True, maximum_training_iterations=100):
        self.maximum_training_iterations = maximum_training_iterations
        t0 = time()
        self.opt.minimize(self.model, disp=verbose, maxiter=self.maximum_training_iterations)
        dt = time() - t0
        if verbose:
            print(Fore.YELLOW + "training time: {}s".format(dt) + Fore.RESET)

    def metadata(self):
        return {
            'N': self.N,
            'D': self.D,
            'P': self.P,
            'M': self.M,
            'maximum_training_iterations': self.maximum_training_iterations,
            'kernel_type': self.model_def['class'].__name__,
            'initial_hyper_params': self.model_def['initial_hyper_params'],
            'initial_likelihood_variance': self.model_def['initial_likelihood_variance'],
        }

    def save(self):
        saver = gpf.saver.Saver()
        log_path = experiments_util.experiment_name('separate_independent', 'gpf')
        full_log_path = os.path.join(os.getcwd(), 'log_data', log_path)
        experiments_util.make_log_dir(full_log_path)
        experiments_util.write_metadata(self.metadata(), log_path)
        model_path = os.path.join(full_log_path, 'model')
        print(Fore.CYAN + "Saving model to {}".format(full_log_path) + Fore.RESET)
        saver.save(model_path, self.model)

    def load(self):
        self.model = gpflow.saver.Saver().load(model_path)
