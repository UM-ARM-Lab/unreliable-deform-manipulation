import numpy as np

import control


class LQRAct:

    def __init__(self, linear_tf_model, max_v):
        self.linear_tf_model = linear_tf_model
        self.max_v = max_v
        _, self.state_matrix, self.control_matrix, _ = self.linear_tf_model.get_ABCD()
        R = np.eye(self.linear_tf_model.M)
        Q = np.eye(self.linear_tf_model.L) * 10
        self.K, S, E = control.lqr(self.state_matrix, self.control_matrix, R, Q)

    def act(self, o, og):
        """ return the action which gives the lowest cost for the predicted next state """
        u = np.dot(-self.K, o - og)
        u = u / np.linalg.norm(u) * self.max_v
        o_next = self.linear_tf_model.simple_predict(o, u.reshape(2, 1))
        return u.reshape(-1, 1, 2), o_next

    def __repr__(self):
        return "max_v: {}".format(self.max_v)
