import numpy as np
import ompl.util as ou
from ompl import base as ob
from ompl import control as oc


class OMPLAct:

    def __init__(self, action_selector, directed_control_sampler, M, L, dt, og, max_v):
        self.action_selector = action_selector
        self.directed_control_sampler = directed_control_sampler
        self.dt = dt
        self.M = M
        self.L = L
        self.og = og
        self.max_v = max_v

        self.latent_space = ob.RealVectorStateSpace(self.M)
        self.latent_space.setBounds(-10, 10)

        self.control_space = oc.RealVectorControlSpace(self.latent_space, self.L)

        self.ss = oc.SimpleSetup(self.control_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(self.dt)

        self.si.setDirectedControlSamplerAllocator(self.directed_control_sampler.allocator(self.action_selector))
        self.directed_control_sampler = self.si.allocDirectedControlSampler()

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
        self.directed_control_sampler.reset()
        start = ob.State(self.latent_space)
        goal = ob.State(self.latent_space)
        for i in range(self.M):
            start[i] = o[i, 0].astype(np.float64)
            goal[i] = self.og[i, 0].astype(np.float64)

        self.ss.clear()
        # the threshold on "cost-to-goal" is interpretable here as euclidian distance
        epsilon = 0.1
        self.ss.setStartAndGoalStates(start, goal, 0.4 * epsilon)
        solved = self.ss.solve(10)
        print("Planning time: {}".format(self.ss.getLastPlanComputationTime()))
        if solved:
            ompl_path = self.ss.getSolutionPath()

            numpy_controls = np.ndarray((ompl_path.getControlCount(), 1, self.L))
            numpy_states = np.ndarray((ompl_path.getStateCount(), self.M))
            for i, state in enumerate(ompl_path.getStates()):
                for j in range(self.M):
                    numpy_states[i, j] = state[j]
            for i, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
                numpy_controls[i, 0, 0] = control[0]
                numpy_controls[i, 0, 1] = control[1]

            # SMOOTHING
            new_states = list(numpy_states)
            new_controls = list(numpy_controls)
            iter = 0
            while iter < 10 and len(new_states) > 2:
                iter += 1
                start_idx = np.random.randint(0, len(new_states) - 1)
                shortcut_start = new_states[start_idx]
                end_idx = np.random.randint(start_idx + 1, len(new_states))
                shortcut_end = new_states[end_idx]

                success, new_shortcut_us, new_shortcut_os = self.directed_control_sampler.multi_act(shortcut_start, shortcut_end)

                if success:
                    # popping changes the indexes of everything, so we just pop tat start_idx the right number of times
                    for i in range(start_idx, end_idx):
                        new_states.pop(start_idx)
                        new_controls.pop(start_idx)
                    for i, (shortcut_u, shortcut_o) in enumerate(zip(new_shortcut_us, new_shortcut_os)):
                        new_states.insert(start_idx + i, np.squeeze(shortcut_o))  # or maybe shortcut_end?
                        new_controls.insert(start_idx + i, shortcut_u.reshape(1, 2))
                else:
                    print("shortcutting failed to progress...")

            numpy_states = np.array(new_states)
            numpy_controls = np.array(new_controls)

            if verbose:
                self.directed_control_sampler.plot_controls(numpy_controls)
                self.directed_control_sampler.plot_2d(o, self.og, numpy_states)
            lengths = [np.linalg.norm(numpy_states[i] - numpy_states[i - 1]) for i in range(1, len(numpy_states))]
            path_length = np.sum(lengths)
            final_error = np.linalg.norm(numpy_states[-1] - self.og)
            duration = self.dt * len(numpy_states)
            print("Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s".format(
                final_error, path_length, len(numpy_states), duration))
            return numpy_controls, numpy_states
        else:
            raise RuntimeError("No Solution found from {} to {}".format(start, goal))
