import numpy as np

import control
from link_bot_agent import action_selector


class DualLQRActionSelector(action_selector.ActionSelector):

    def __init__(self, linear_constraint_model, max_v):
        super(DualLQRActionSelector, self).__init__()
        self.linear_constraint_model = linear_constraint_model
        self.max_v = max_v
        self.A_d, self.B_d, self.A_k, self.B_k = self.linear_constraint_model.get_dynamics_matrices()
        Q = np.eye(self.linear_constraint_model.M)
        # apparently if this R is too small things explode???
        R = np.eye(self.linear_constraint_model.L) * 0.1
        # control is based on the dynamics (hence the letter d) state
        self.K, S, E = control.lqr(self.A_d, self.B_d, Q, R)

    def just_d_act(self, o_d, o_d_goal):
        """ return the action which gives the lowest cost for the predicted next state """
        u = np.dot(-self.K, o_d - o_d_goal)

        u_norm = np.linalg.norm(u)
        if u_norm > 1e-9:
            if u_norm > self.max_v:
                scaling = self.max_v
            else:
                scaling = u_norm
            u = u * scaling / u_norm

        o_d_next = self.linear_constraint_model.simple_predict(o_d, u.reshape(2, 1))
        return u.reshape(-1, 1, 2), o_d_next

    def dual_act(self, o_d, o_k, o_d_goal):
        """ return the action which gives the lowest cost for the predicted next state """
        # the state g
        u = np.dot(-self.K, o_d - o_d_goal)

        u_norm = np.linalg.norm(u)
        if u_norm > 1e-9:
            if u_norm > self.max_v:
                scaling = self.max_v
            else:
                scaling = u_norm
            u = u * scaling / u_norm

        o_d_next, o_k_next = self.linear_constraint_model.simple_dual_predict(o_d, o_k, u.reshape(2, 1))
        return u.reshape(-1, 1, 2), o_d_next, o_k_next
