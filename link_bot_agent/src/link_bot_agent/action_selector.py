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
        for _ in range(100):
            u, o_next = self.act(o, og)
            if np.linalg.norm(u) < 1e-3:
                return False, None, None

            if np.allclose(o, o_next, rtol=0.01):
                break

            us.append(u)
            os.append(o_next)

            o = o_next

        return True, np.array(us), np.array(os)
