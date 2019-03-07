import numpy as np
import ompl.util as ou
from ompl import base as ob
from ompl import control as oc
from link_bot_agent.gurobi_directed_control_sampler import GurobiDirectedControlSampler
from link_bot_agent.random_directed_control_sampler import RandomDirectedControlSampler
from link_bot_agent.lqr_directed_control_sampler import LQRDirectedControlSampler


class OMPLAct:

    def __init__(self, gurobi_solver, og, max_v):
        self.linear_tf_model = gurobi_solver.linear_tf_model
        self.gurobi_solver = gurobi_solver
        self.dt = self.linear_tf_model.dt
        self.M = self.linear_tf_model.M
        self.L = self.linear_tf_model.L
        self.og = og
        self.max_v = max_v

        self.latent_space = ob.RealVectorStateSpace(self.M)
        self.latent_space.setBounds(-10, 10)

        self.control_space = oc.RealVectorControlSpace(self.latent_space, self.L)
        # The OMPL control space will be direction and magnitude, not vx, vy directly
        # becahse otherwise we cannot constrain the velocity correctly
        self.control_bounds = ob.RealVectorBounds(2)

        self.ss = oc.SimpleSetup(self.control_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        # self.ss.setStatePropagator(oc.StatePropagatorFn(self.dumb_propagate))
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        self.si = self.ss.getSpaceInformation()
        self.si.setMinMaxControlDuration(1, 50)
        self.si.setPropagationStepSize(self.linear_tf_model.dt)

        self.MyDirectedControlSampler = LQRDirectedControlSampler
        self.si.setDirectedControlSamplerAllocator(self.MyDirectedControlSampler.allocator(self.linear_tf_model, max_v))

        # self.MyDirectedControlSampler = GurobiDirectedControlSampler
        # self.si.setDirectedControlSamplerAllocator(self.MyDirectedControlSampler.allocator(self.gurobi_solver))

        # self.MyDirectedControlSampler = RandomDirectedControlSampler
        # self.si.setDirectedControlSamplerAllocator(self.MyDirectedControlSampler.allocator(self.linear_tf_model))

        self.planner = oc.RRT(self.si)
        self.ss.setPlanner(self.planner)

    def setSeed(self, seed):
        ou.RNG.setSeed(seed)

    def isStateValid(self, state):
        # perform collision checking or check if other constraints are
        # satisfied
        return self.latent_space.satisfiesBounds(state)

    def propagate(self, start, control, duration, state):
        _, B, C, _ = self.linear_tf_model.get_ABCD()
        u = np.array([control[0], control[1]])
        o = np.array([start[i] for i in range(self.M)])
        o_next = o + duration * np.matmul(B, o) + duration * np.matmul(C, u)
        # modify to do the propagation
        for i in range(self.M):
            state[i] = o_next[i].astype(np.float64)

    def dumb_propagate(self, start, control, duration, state):
        print("DUMB PROPAGATE")
        state[0] = start[0] + duration * control[0]
        state[1] = start[1] + duration * control[1]

    def act(self, o, verbose=False):
        """ return the action which gives the lowest cost for the predicted next state """
        self.MyDirectedControlSampler.reset()
        start = ob.State(self.latent_space)
        start[0] = o[0, 0].astype(np.float64)
        start[1] = o[1, 0].astype(np.float64)

        goal = ob.State(self.latent_space)
        goal[0] = self.og[0, 0].astype(np.float64)
        goal[1] = self.og[1, 0].astype(np.float64)

        self.ss.clear()
        # TODO: How do we compute epsilon in latent space???
        self.ss.setStartAndGoalStates(start, goal, 0.01)
        solved = self.ss.solve(0.1)
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
                numpy_controls[i, 0, 0] = control[0]
                numpy_controls[i, 0, 1] = control[1]

            # SMOOTHING
            # new_states = list(numpy_states)
            # new_controls = list(numpy_controls)
            # new_durations = list(durations)
            # iter = 0
            # while iter < 100 and len(new_states) > 2:
            #     iter += 1
            #     start_idx = np.random.randint(0, len(new_states))
            #     shortcut_start = new_states[start_idx]
            #     end_idx = np.random.uniform(start_idx, len(new_states) - 1)
            #     end_idx_floor = np.floor(end_idx).astype(np.int32)
            #     end_idx_ceil = np.ceil(end_idx).astype(np.int32)
            #     if start_idx == end_idx_floor:
            #         continue
            #     floor_point = new_states[end_idx_floor]
            #     ceil_point = new_states[end_idx_ceil]
            #     # linearly interpolate in latent space and try to make a shortcut to this point
            #     shortcut_end = floor_point + (end_idx - end_idx_floor) * (ceil_point - floor_point) / (
            #             end_idx_ceil - end_idx_floor)
            #     shortcut_start = np.expand_dims(shortcut_start, axis=1)
            #     shortcut_end = np.expand_dims(shortcut_end, axis=1)
            #     new_shortcut_us, new_shortcut_os = self.gurobi_solver.multi_act(shortcut_start, shortcut_end)
            #     if np.allclose(new_shortcut_os[-1], shortcut_end, rtol=0.01):
            #         # popping changes the indexes of everything, so we just pop tat start_idx the right number of times
            #         for i in range(start_idx, end_idx_ceil):
            #             new_states.pop(start_idx)
            #             new_controls.pop(start_idx)
            #             new_durations.pop(start_idx)
            #         for i, (shortcut_u, shortcut_o) in enumerate(zip(new_shortcut_us, new_shortcut_os)):
            #             new_states.insert(start_idx + i, np.squeeze(shortcut_o))  # or maybe shortcut_end?
            #             new_controls.insert(start_idx + i, np.expand_dims(shortcut_u, axis=0))
            #             new_durations.insert(start_idx + i, self.dt)

            # numpy_states = np.array(new_states)
            # numpy_controls = np.array(new_controls)
            # durations = np.array(new_durations)

            if verbose:
                self.MyDirectedControlSampler.plot(o, self.og, numpy_states)
                lengths = [np.linalg.norm(numpy_states[i] - numpy_states[i - 1]) for i in range(1, len(numpy_states))]
                path_length = np.sum(lengths)
                final_error = np.linalg.norm(numpy_states[-1] - self.og)
                duration = np.sum(durations)
                print("Final Error: {:0.4f}m, Path Length: {:0.4f}m, Steps {}, Duration: {:0.2f}s".format(
                    final_error, path_length, len(durations), duration))
            return numpy_controls, durations
        else:
            raise RuntimeError("No Solution found from {} to {}".format(start, goal))
