import numpy as np
import matplotlib.pyplot as plt
import ompl.util as ou
from ompl import base as ob
from ompl import control as oc


class OMPLAct:

    def __init__(self, tf_model, action_selector, directed_control_sampler, dt, o_goal, max_v):
        self.tf_model = tf_model
        self.action_selector = action_selector
        self.directed_control_sampler = directed_control_sampler
        self.dt = dt
        self.L = self.tf_model.L
        self.M = self.tf_model.M
        self.N = self.tf_model.N
        self.P = self.tf_model.P
        self.Q = self.tf_model.Q
        self.o_d_goal = o_goal
        self.max_v = max_v

        dynamics_latent_space = ob.RealVectorStateSpace(self.M)
        constraint_latent_space = ob.RealVectorStateSpace(self.P)
        # TODO: these are arbitrary
        dynamics_latent_space.setName("dynamics latent space")
        dynamics_latent_space.setBounds(-5, 5)
        constraint_latent_space.setName("constraint latent space")
        constraint_latent_space.setBounds(-5, 5)

        self.latent_space = ob.CompoundStateSpace()
        self.latent_space.addSubspace(dynamics_latent_space, 1)
        self.latent_space.addSubspace(constraint_latent_space, 0)

        self.control_space = oc.RealVectorControlSpace(self.latent_space, self.L)

        self.ss = oc.SimpleSetup(self.control_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(self.dt)
        self.si.setMinMaxControlDuration(1, 10)

        self.si.setDirectedControlSamplerAllocator(self.directed_control_sampler.allocator(self.action_selector))
        self.directed_control_sampler = self.si.allocDirectedControlSampler()

        self.planner = oc.RRT(self.si)
        self.planner.setIntermediateStates(False)
        # self.planner.setGoalBias(0.99)
        self.ss.setPlanner(self.planner)

    def setSeed(self, seed):
        ou.RNG.setSeed(seed)

    def isStateValid(self, state):
        # check if our model predicts that constraints are violated in this state
        constraint_state = np.ndarray(self.P)
        for i in range(self.P):
            constraint_state[i] = state[1][i]
        constraint_violated = self.tf_model.constraint_violated(constraint_state)
        constraint_violated = np.any(constraint_violated)
        valid = self.latent_space.satisfiesBounds(state) and not constraint_violated
        return valid

    def propagate(self, start, control, duration, state):
        # apparently when we used a directed control sampler this function is not used.
        raise NotImplementedError("Because we used custom directed control sampler, we expect propagate is not used")

    def act(self, sdf, o_d_start, o_k_start, verbose=False):
        # create start and goal states
        self.directed_control_sampler.reset()
        start = ob.State(self.latent_space)
        goal = ob.State(self.latent_space)
        for i in range(self.M):
            start()[0][i] = o_d_start[i, 0].astype(np.float64)
            goal()[0][i] = self.o_d_goal[i, 0].astype(np.float64)
        for i in range(self.P):
            start()[1][i] = o_k_start[i, 0].astype(np.float64)
            goal()[1][i] = -999

        self.ss.clear()
        # the threshold on "cost-to-goal" is interpretable here as Euclidian distance
        epsilon = 0.1
        self.ss.setStartAndGoalStates(start, goal, 0.4 * epsilon)
        solved = self.ss.solve(500)
        print("Planning time: {}".format(self.ss.getLastPlanComputationTime()))
        if solved:
            ompl_path = self.ss.getSolutionPath()

            numpy_controls = np.ndarray((ompl_path.getControlCount(), 1, self.L))
            numpy_d_states = np.ndarray((ompl_path.getStateCount(), self.M))
            numpy_k_states = np.ndarray((ompl_path.getStateCount(), self.P))
            for i, state in enumerate(ompl_path.getStates()):
                for j in range(self.M):
                    numpy_d_states[i, j] = state[0][j]
                for j in range(self.P):
                    numpy_k_states[i, j] = state[1][j]
            for i, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
                numpy_controls[i, 0, 0] = control[0]
                numpy_controls[i, 0, 1] = control[1]

            # SMOOTHING
            # new_states = list(numpy_states)
            # new_controls = list(numpy_controls)
            # shortcut_iter = 0
            # shortcut_successes = 0
            # while shortcut_iter < 200 and len(new_states) > 2:
            #     shortcut_iter += 1
            #     start_idx = np.random.randint(0, len(new_states) - 1)
            #     shortcut_start = new_states[start_idx]
            #     end_idx = np.random.randint(start_idx + 1, len(new_states))
            #     shortcut_end = new_states[end_idx]
            #
            #     success, new_shortcut_us, new_shortcut_os = self.directed_control_sampler.dual_multi_act(shortcut_start,
            #                                                                                              shortcut_end)
            #
            #     if success:
            #         # popping changes the indexes of everything, so we just pop tat start_idx the right number of times
            #         for i in range(start_idx, end_idx):
            #             new_states.pop(start_idx)
            #             new_controls.pop(start_idx)
            #         for i, (shortcut_u, shortcut_o) in enumerate(zip(new_shortcut_us, new_shortcut_os)):
            #             new_states.insert(start_idx + i, np.squeeze(shortcut_o))  # or maybe shortcut_end?
            #             new_controls.insert(start_idx + i, shortcut_u.reshape(1, 2))
            #         shortcut_successes += 1
            #
            # numpy_states = np.array(new_states)
            # numpy_controls = np.array(new_controls)

            if verbose:
                # print("{}/{} shortcuts succeeded".format(shortcut_successes, shortcut_iter))
                self.directed_control_sampler.plot_dual_sdf(sdf, o_d_start, self.o_d_goal, numpy_d_states, numpy_k_states)
                plt.show()
            lengths = [np.linalg.norm(numpy_d_states[i] - numpy_d_states[i - 1]) for i in range(1, len(numpy_d_states))]
            path_length = np.sum(lengths)
            final_error = np.linalg.norm(numpy_d_states[-1] - self.o_d_goal)
            duration = self.dt * len(numpy_d_states)
            print("Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s".format(
                final_error, path_length, len(numpy_d_states), duration))
            return numpy_controls, numpy_d_states
        else:
            raise RuntimeError("No Solution found from {} to {}".format(start, goal))
