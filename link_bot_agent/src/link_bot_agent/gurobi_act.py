import gurobipy as gurobi
import numpy as np


class GurobiAct:

    def __init__(self, linear_tf_model, og, max_v):
        self.linear_tf_model = linear_tf_model
        self.og = og
        self.max_v = max_v
        self.gurobi_model = gurobi.Model("model")
        self.gurobi_model.setParam('OutputFlag', 0)
        self.u1 = self.gurobi_model.addVar(name="u1", lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
        self.u2 = self.gurobi_model.addVar(name="u2", lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
        self.u = np.array([[self.u1], [self.u2]])
        self.gurobi_model.addQConstr(self.u1 * self.u1 + self.u2 * self.u2 <= max_v ** 2, "c0")
        self.A, self.B, self.C, self.D = self.linear_tf_model.get_ABCD()

    def act(self, o):
        """ return the action which gives the lowest cost for the predicted next state """
        o_next = o + self.linear_tf_model.dt * np.dot(self.B, o) + self.linear_tf_model.dt * np.dot(self.C, self.u)
        distance = np.squeeze(self.og - o_next)
        obj = np.dot(np.dot(distance, self.D), distance.T)
        self.gurobi_model.setObjective(obj, gurobi.GRB.MINIMIZE)

        self.gurobi_model.optimize()
        u = np.array([v.x for v in self.gurobi_model.getVars()]).reshape(1, 1, 2)
        return u

    def __repr__(self):
        return "og: {}, max_v: {}".format(np.array2string(self.og.T), self.max_v)
