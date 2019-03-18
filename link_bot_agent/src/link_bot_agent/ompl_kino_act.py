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

        self.ss = oc.SimpleSetup(self.control_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(self.linear_tf_model.dt)

        # self.MyDirectedControlSampler = LQRDirectedControlSampler
        # self.si.setDirectedControlSamplerAllocator(self.MyDirectedControlSampler.allocator(self.linear_tf_model, max_v))

        self.MyDirectedControlSampler = GurobiDirectedControlSampler
        self.si.setDirectedControlSamplerAllocator(self.MyDirectedControlSampler.allocator(self.gurobi_solver))

        # self.MyDirectedControlSampler = RandomDirectedControlSampler
        # self.si.setDirectedControlSamplerAllocator(self.MyDirectedControlSampler.allocator(self.linear_tf_model))

        self.planner = oc.RRT(self.si)
        self.planner.setGoalBias(0.5)
        self.ss.setPlanner(self.planner)

    def setSeed(self, seed):
        ou.RNG.setSeed(seed)

    def isStateValid(self, state):
        # perform collision checking or check if other constraints are
        # satisfied
        return self.latent_space.satisfiesBounds(state)

    def propagate(self, start, control, duration, state):
        # apprently when we used a directed control sampler this function is not used.
        pass

    def act(self, o, verbose=False):
        """ return the action which gives the lowest cost for the predicted next state """
        self.MyDirectedControlSampler.reset()
        start = ob.State(self.latent_space)
        goal = ob.State(self.latent_space)
        for i in range(self.M):
            start[i] = o[i, 0].astype(np.float64)
            goal[i] = self.og[i, 0].astype(np.float64)

        self.ss.clear()
        # the threshold on "cost-to-goal" is interpretable here as euclidian distance
        epsilon = 0.1
        self.ss.setStartAndGoalStates(start, goal, 0.4 * epsilon)
        solved = self.ss.solve(30)
        print("Planning time: {}".format(self.ss.getLastPlanComputationTime()))
        print("Number of nodes sampled Nodes: {}".format(GurobiDirectedControlSampler.num_samples))
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
            new_states = list(numpy_states)
            new_controls = list(numpy_controls)
            new_durations = list(durations)
            iter = 0
            while iter < 10 and len(new_states) > 2:
                iter += 1
                start_idx = np.random.randint(0, len(new_states) - 1)
                shortcut_start = new_states[start_idx]
                end_idx = np.random.randint(start_idx + 1, len(new_states))
                shortcut_end = new_states[end_idx]

                success, new_shortcut_us, new_shortcut_os = self.gurobi_solver.multi_act(shortcut_start, shortcut_end)

                if success:
                    # popping changes the indexes of everything, so we just pop tat start_idx the right number of times
                    for i in range(start_idx, end_idx):
                        new_states.pop(start_idx)
                        new_controls.pop(start_idx)
                        new_durations.pop(start_idx)
                    for i, (shortcut_u, shortcut_o) in enumerate(zip(new_shortcut_us, new_shortcut_os)):
                        new_states.insert(start_idx + i, np.squeeze(shortcut_o))  # or maybe shortcut_end?
                        new_controls.insert(start_idx + i, shortcut_u.reshape(1, 2))
                        new_durations.insert(start_idx + i, self.dt)
                else:
                    print("shortcutting failed to progress...")

            numpy_states = np.array(new_states)
            numpy_controls = np.array(new_controls)
            durations = np.array(new_durations)

            if verbose:
                self.MyDirectedControlSampler.plot_controls(numpy_controls)
                self.MyDirectedControlSampler.plot_2d(o, self.og, numpy_states)
            lengths = [np.linalg.norm(numpy_states[i] - numpy_states[i - 1]) for i in range(1, len(numpy_states))]
            path_length = np.sum(lengths)
            final_error = np.linalg.norm(numpy_states[-1] - self.og)
            duration = np.sum(durations)
            print("Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s".format(
                final_error, path_length, len(durations), duration))
            return numpy_controls, durations
        else:
            raise RuntimeError("No Solution found from {} to {}".format(start, goal))
