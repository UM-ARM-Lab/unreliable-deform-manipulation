import numpy as np

import gurobipy as gurobi


class ActionSelector(object):

    def __init__(self):
        pass

    def act(self, o, og):
        raise NotImplementedError("ActionSelector is an abstract class.")

    def multi_act(self, o, og):
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
