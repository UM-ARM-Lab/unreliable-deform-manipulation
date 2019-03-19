import numpy as np

import gurobipy as gurobi


class OneStepGurobiAct:

    def __init__(self, linear_tf_model, max_v):
        self.linear_tf_model = linear_tf_model
        self.max_v = max_v
        self.gurobi_model = gurobi.Model("model")
        self.gurobi_model.setParam('OutputFlag', 0)
        self.u1 = self.gurobi_model.addVar(name="u1", lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
        self.u2 = self.gurobi_model.addVar(name="u2", lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
        self.gurobi_u = np.array([[self.u1], [self.u2]])
        self.gurobi_model.addQConstr(self.u1 * self.u1 + self.u2 * self.u2 <= max_v ** 2, "c0")
        self.A, self.B, self.C, self.D = self.linear_tf_model.get_ABCD()

    def act(self, o, og):
        """ return the action which gives the lowest cost for the predicted next state """
        gurobi_o_next = self.linear_tf_model.simple_predict(o, self.gurobi_u)
        distance = np.squeeze(og - gurobi_o_next)
        obj = np.dot(np.dot(distance, self.D), distance.T)
        self.gurobi_model.setObjective(obj, gurobi.GRB.MINIMIZE)

        self.gurobi_model.optimize()
        numpy_u = np.array([v.x for v in self.gurobi_model.getVars()])
        o_next = self.linear_tf_model.simple_predict(o, numpy_u.reshape(2, 1))
        return numpy_u.reshape(1, 1, 2), o_next

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

    def __repr__(self):
        return "max_v: {}".format(self.max_v)
