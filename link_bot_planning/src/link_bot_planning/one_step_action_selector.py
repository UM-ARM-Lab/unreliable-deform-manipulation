import numpy as np

import gurobi
from link_bot_planning import action_selector


class OneStepGurobiAct(action_selector.ActionSelector):

    def __init__(self, tf_model, max_v):
        super(OneStepGurobiAct, self).__init__()
        self.tf_model = tf_model
        self.max_v = max_v
        self.gurobi_model = gurobi.Model("model")
        self.gurobi_model.setParam('OutputFlag', 0)
        self.u1 = self.gurobi_model.addVar(name="u1", lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
        self.u2 = self.gurobi_model.addVar(name="u2", lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
        self.gurobi_u = np.array([[self.u1], [self.u2]])
        self.gurobi_model.addQConstr(self.u1 * self.u1 + self.u2 * self.u2 <= max_v ** 2, "c0")
        self.A, self.B, self.C, self.D = self.tf_model.get_ABCD()

    def just_d_act(self, o_d, o_d_goal):
        """ return the action which gives the lowest cost for the predicted next state """
        gurobi_o_next = self.tf_model.simple_predict(o_d, self.gurobi_u)
        distance = np.squeeze(o_d_goal - gurobi_o_next)
        obj = np.dot(np.dot(distance, self.D), distance.T)
        self.gurobi_model.setObjective(obj, gurobi.GRB.MINIMIZE)

        self.gurobi_model.optimize()
        numpy_u = np.array([v.x for v in self.gurobi_model.getVars()])
        o_next = self.tf_model.simple_predict(o_d, numpy_u.reshape(2, 1))
        return numpy_u.reshape(1, 1, 2), o_next
