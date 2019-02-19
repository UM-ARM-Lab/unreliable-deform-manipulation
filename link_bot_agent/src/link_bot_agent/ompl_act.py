from ompl import base as ob
import matplotlib.pyplot as plt
from ompl import control as oc
import numpy as np


class OMPLAct:

    def __init__(self, linear_tf_model, og, max_v):
        self.linear_tf_model = linear_tf_model
        self.dt = self.linear_tf_model.dt
        self.M = self.linear_tf_model.M
        self.L = self.linear_tf_model.L
        self.og = og
        self.max_v = max_v

        self.latent_space = ob.RealVectorStateSpace(self.M)
        self.latent_space.setBounds(-5, 5)

        self.control_space = oc.RealVectorControlSpace(self.latent_space, self.L)
        # The OMPL control space will be direction and magnitude, not vx, vy directly
        # becahse otherwise we cannot constrain the velocity correctly
        self.control_bounds = ob.RealVectorBounds(2)
        # angle
        self.control_bounds.setLow(0, -np.pi)
        self.control_bounds.setHigh(0, np.pi)
        # speed
        self.control_bounds.setLow(1, 0)
        self.control_bounds.setHigh(1, max_v)
        self.control_space.setBounds(self.control_bounds)

        self.ss = oc.SimpleSetup(self.control_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))
        self.si = self.ss.getSpaceInformation()
        self.planner = oc.RRT(self.si)
        self.ss.setPlanner(self.planner)
        self.points = []

    def isStateValid(self, state):
        # perform collision checking or check if other constraints are
        # satisfied
        return self.latent_space.satisfiesBounds(state)

    def propagate(self, start, control, duration, state):
        _, B, C, _ = self.linear_tf_model.get_ABCD()
        u = np.array([control[i] for i in range(self.L)])
        o = np.array([start[i] for i in range(self.M)])
        o_next = o + duration * np.matmul(B, o) + duration * np.matmul(C, u)
        # modify to do the propagation
        self.points.append(o_next)
        for i in range(self.M):
            state[i] = o_next[i].astype(np.float64)


    def act(self, o):
        """ return the action which gives the lowest cost for the predicted next state """
        start = ob.State(self.latent_space)
        start[0] = o[0, 0].astype(np.float64)
        start[1] = o[1, 0].astype(np.float64)

        goal = ob.State(self.latent_space)
        goal[0] = self.og[0, 0].astype(np.float64)
        goal[1] = self.og[1, 0].astype(np.float64)
        self.points = []

        # TODO: How do we compute epsilon in latent space???
        epsilon = 0.01
        self.ss.setStartAndGoalStates(start, goal, epsilon)
        solved = self.ss.solve(10.0)
        if solved:
            ompl_path = self.ss.getSolutionPath()
            controls = ompl_path.getControls()
            control_durations = ompl_path.getControlsDurations()
            state = ompl_path.getStates()

            numpy_path = np.ndarray((ompl_path.getxeCount(), 1, self.L))
            for i, control in enumerate(ompl_path.getStates()):
                numpy_path[i, 0, 0] = control[0]
                numpy_path[i, 0, 1] = control[1]
            plt.scatter(o)
            plt.scatter(og)
            plt.plot(self.points)
            u = numpy_path.squeeze(axis=1)
            plt.quiver(u[:, 0], u[:, 1])
            plt.axis('equal')
            plt.show()
            return numpy_path
        else:
            raise RuntimeError("No Solution found from {} to {}".format(start, goal))
