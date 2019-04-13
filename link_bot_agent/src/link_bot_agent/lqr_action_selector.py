import numpy as np

import control
from link_bot_agent import action_selector


class LQRActionSelector(action_selector.ActionSelector):

    def __init__(self, linear_tf_model, max_v):
        super(LQRActionSelector, self).__init__()
        self.linear_tf_model = linear_tf_model
        self.max_v = max_v
        self.state_matrix, self.control_matrix = self.linear_tf_model.get_dynamics_matrices()
        Q = np.eye(self.linear_tf_model.M)
        # apparently if this R is too small things explode???
        R = np.eye(self.linear_tf_model.L) * 1e-9
        self.K, S, E = control.lqr(self.state_matrix, self.control_matrix, Q, R)

    def act(self, o, og):
        """ return the action which gives the lowest cost for the predicted next state """
        u = np.dot(-self.K, o - og)
        o_next = self.linear_tf_model.simple_predict(o, u.reshape(2, 1))
        return u.reshape(-1, 1, 2), o_next
