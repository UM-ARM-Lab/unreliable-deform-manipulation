import numpy as np

import gurobipy as gurobi


class ActionSelector(object):

    def __init__(self):
        pass

    def just_d_act(self, o_d, o_d_goal):
        raise NotImplementedError("ActionSelector is an abstract class.")

    def dual_act(self, o_d, o_k, o_d_goal):
        raise NotImplementedError("ActionSelector is an abstract class.")

    def just_d_multi_act(self, o, og):
        """ return the action which gives the lowest cost for the predicted next state """
        o = o.reshape(-1, 1)
        og = og.reshape(-1, 1)
        us = []
        os = [o]
        errors = []
        for _ in range(100):
            u, o_next = self.act(o, og)
            if np.linalg.norm(u) < 1e-3:
                return False, None, None

            us.append(u)
            os.append(o_next)

            error = np.linalg.norm(o - og)
            errors.append(error)
            if error < 0.05:
                return True, np.array(us), np.array(os)
            o = o_next

        return False, None, None

    def dual_multi_act(self, o_d, o_k, o_d_goal):
        us = []
        o_ks = [o_d]
        o_ds = [o_k]
        errors = []
        for _ in range(100):
            u, o_d_next, o_k_next = self.dual_act(o_d, o_k, o_d_goal)
            if np.linalg.norm(u) < 1e-3:
                return False, None, None, None

            us.append(u)
            o_ds.append(o_d)
            o_ks.append(o_k)

            error = np.linalg.norm(o_d - o_d_goal)
            errors.append(error)
            if error < 0.05:
                return True, np.array(us), np.array(o_ds), np.array(o_ks)
            o_d = o_d_next
            o_k = o_k_next

        return False, None, None, None
