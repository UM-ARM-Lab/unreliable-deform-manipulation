import numpy as np
import tensorflow as tf
import os
from colorama import Fore


class BaseModel:

    def __init__(self, args_dict, N):
        """
        args_dict: the argsparse args but as a dict
        N: dimensionality of the full state
        """
        self.args_dict = args_dict
        self.N = N

        self.seed = self.args_dict['seed']
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)

    def setup(self):
        if self.args['checkpoint']:
            self.sess.run([tf.local_variables_initializer()])
            self.load()
        else:
            self.init()

    def init(self):
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def load(self):
        self.saver.restore(self.sess, self.args['checkpoint'])
        global_step = self.sess.run(self.global_step)
        print(Fore.CYAN + "Restored ckpt {} at step {:d}".format(self.args['checkpoint'], global_step) + Fore.RESET)

    def save(self, log_path, log=True, loss=None):
        global_step = self.sess.run(self.global_step)
        if log:
            if loss is not None:
                print(Fore.CYAN + "Saving ckpt {} at step {:d} with loss {}".format(log_path, global_step,
                                                                                    loss) + Fore.RESET)
            else:
                print(Fore.CYAN + "Saving ckpt {} at step {:d}".format(log_path, global_step) + Fore.RESET)
        self.saver.save(self.sess, os.path.join(log_path, "nn.ckpt"), global_step=self.global_step)

    def __str__(self):
        return "base_model"
