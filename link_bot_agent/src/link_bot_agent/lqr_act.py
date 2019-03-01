import control
import numpy as np


class GurobiAct:

    def __init__(self, linear_tf_model, max_v):
        self.linear_tf_model = linear_tf_model
        self.max_v = max_v
        _, self.state_matrix, self.control_matrix, _ = self.linear_tf_model.get_ABCD()

    def act(self, o, og):
        """ return the action which gives the lowest cost for the predicted next state """
        K, S, E = control.lqr(self.state_matrix, self.control_matrix)
        return np.dot(-K, o - og)

    def __repr__(self):
        return "max_v: {}".format(self.max_v)
