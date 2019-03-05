import numpy as np
import ompl.util as ou
import ompl.geometric as og
from ompl import base as ob
import control


class OMPLAct:

    def __init__(self, gurobi_solver, o_g, max_v):
        self.linear_tf_model = gurobi_solver.linear_tf_model
        self.gurobi_solver = gurobi_solver
        self.dt = self.linear_tf_model.dt
        self.M = self.linear_tf_model.M
        self.L = self.linear_tf_model.L
        self.o_g = o_g
        self.max_v = max_v

        self.latent_space = ob.RealVectorStateSpace(self.M)
        self.latent_space.setBounds(-5, 5)

        self.ss = og.SimpleSetup(self.latent_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.si = self.ss.getSpaceInformation()

        self.planner = og.RRT(self.si)
        self.ss.setPlanner(self.planner)

    def setSeed(self, seed):
        ou.RNG.setSeed(seed)

    def isStateValid(self, state):
        return self.latent_space.satisfiesBounds(state)

    def steer(self, states):
        _, state_matrix, control_matrix, _ = self.linear_tf_model.get_ABCD()
        Q = np.eye(self.M)
        R = np.eye(self.L)
        K, S, E = control.lqr(state_matrix, control_matrix, Q, R)
        print(E)
        n = len(states)
        controls = np.ndarray((n - 1, 1, 2))
        for i in range(n - 1):
            u = np.dot(-K, (states[i] - states[i + 1]))
            controls[i] = u

        durations = np.ones(n-1) * self.dt
        return controls, durations

    def act(self, o):
        """ return the action which gives the lowest cost for the predicted next state """
        # self.MyDirectedControlSampler.reset()
        start = ob.State(self.latent_space)
        start[0] = o[0, 0].astype(np.float64)
        start[1] = o[1, 0].astype(np.float64)

        goal = ob.State(self.latent_space)
        goal[0] = self.o_g[0, 0].astype(np.float64)
        goal[1] = self.o_g[1, 0].astype(np.float64)

        self.ss.clear()
        self.ss.setStartAndGoalStates(start, goal, 0.01)
        solved = self.ss.solve(5.0)
        if solved:
            self.ss.simplifySolution()
            ompl_path = self.ss.getSolutionPath()

            numpy_states = np.ndarray((ompl_path.getStateCount(), self.M))
            for i, state in enumerate(ompl_path.getStates()):
                for j in range(self.M):
                    numpy_states[i, j] = state[j]

            lengths = [np.linalg.norm(numpy_states[i] - numpy_states[i - 1]) for i in range(1, len(numpy_states))]
            path_length = np.sum(lengths)
            final_error = np.linalg.norm(numpy_states[-1] - self.o_g)
            print("Final Error: {:0.4f}m, Path Length: {:0.4f}m, Steps {}".format(final_error,
                                                                                  path_length,
                                                                                  len(numpy_states)))
            # convert states into controls and durations
            return self.steer(numpy_states)
        else:
            raise RuntimeError("No Solution found from {} to {}".format(start, goal))
