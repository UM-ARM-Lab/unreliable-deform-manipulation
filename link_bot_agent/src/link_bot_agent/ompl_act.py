from ompl import base as ob
import matplotlib.pyplot as plt
from ompl import control as oc
from ompl.util import RNG
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
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.dumb_propagate))
        # self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))
        self.si = self.ss.getSpaceInformation()
        self.si.setMinMaxControlDuration(1, 50)
        self.si.setPropagationStepSize(0.1)
        self.planner = oc.RRT(self.si)
        self.ss.setPlanner(self.planner)

    def setSeed(self, seed):
        RNG.setSeed(seed)

    def isStateValid(self, state):
        # perform collision checking or check if other constraints are
        # satisfied
        return self.latent_space.satisfiesBounds(state)

    def propagate(self, start, control, duration, state):
        _, B, C, _ = self.linear_tf_model.get_ABCD()
        angle = control[0]
        speed = control[1]
        vx = np.cos(angle) * speed
        vy = np.sin(angle) * speed
        u = np.array([vx, vy])
        o = np.array([start[i] for i in range(self.M)])
        o_next = o + duration * np.matmul(B, o) + duration * np.matmul(C, u)
        # modify to do the propagation
        for i in range(self.M):
            state[i] = o_next[i].astype(np.float64)

    def dumb_propagate(self, start, control, duration, state):
        angle = control[0]
        speed = control[1]
        vx = np.cos(angle) * speed
        vy = np.sin(angle) * speed
        state[0] = start[0] + duration * vx
        state[1] = start[1] + duration * vy

    def act(self, o):
        """ return the action which gives the lowest cost for the predicted next state """
        start = ob.State(self.latent_space)
        start[0] = o[0, 0].astype(np.float64)
        start[1] = o[1, 0].astype(np.float64)

        goal = ob.State(self.latent_space)
        goal[0] = o[0, 0].astype(np.float64)
        goal[1] = o[1, 0].astype(np.float64)

        # TODO: How do we compute epsilon in latent space???
        self.ss.clear()
        self.ss.setStartAndGoalStates(start, goal, 0.01)
        solved = self.ss.solve(5.0)
        if solved:
            ompl_path = self.ss.getSolutionPath()

            numpy_controls = np.ndarray((ompl_path.getControlCount(), 1, self.L))
            durations = np.ndarray(ompl_path.getControlCount())
            numpy_states = np.ndarray((ompl_path.getStateCount(), self.M))
            for i, state in enumerate(ompl_path.getStates()):
                for j in range(self.M):
                    numpy_states[i, j] = state[j]
            for i, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
                durations[i] = duration
                angle = control[0]
                speed = control[1]
                numpy_controls[i, 0, 0] = np.cos(angle) * speed
                numpy_controls[i, 0, 1] = np.sin(angle) * speed

            plt.scatter(o[0, 0], o[1, 0], s=100, label='start')
            plt.scatter(self.og[0, 0], self.og[1, 0], s=100, label='goal')
            plt.plot(numpy_states[:, 0], numpy_states[:, 1])
            u = numpy_controls.squeeze(axis=1)
            plt.quiver(numpy_states[:,0], numpy_states[:, 1], durations * u[:, 0], durations * u[:, 1])
            print(durations, np.linalg.norm(u, axis=1))
            plt.axis('equal')
            plt.legend()
            plt.show()
            return numpy_controls, durations
        else:
            raise RuntimeError("No Solution found from {} to {}".format(start, goal))
