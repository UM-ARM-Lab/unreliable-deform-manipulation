from ompl import base as ob
from ompl import control as oc

import control
import matplotlib.pyplot as plt
import numpy as np


class OMPLAct:

    def __init__(self, tf_model, action_selector, directed_control_sampler, dt, max_v, planner_timeout):
        self.tf_model = tf_model
        self.action_selector = action_selector
        self.directed_control_sampler = directed_control_sampler
        self.dt = dt
        self.planner_timeout = planner_timeout
        self.L = self.tf_model.L
        self.M = self.tf_model.M
        self.N = self.tf_model.N
        self.P = self.tf_model.P
        self.Q = self.tf_model.Q
        self.max_v = max_v

        dynamics_latent_space = ob.RealVectorStateSpace(self.M)
        constraint_latent_space = ob.RealVectorStateSpace(self.P)
        self.arena_size = 10
        # TODO: these are arbitrary
        dynamics_latent_space.setName("dynamics latent space")
        dynamics_latent_space.setBounds(-self.arena_size, self.arena_size)
        constraint_latent_space.setName("constraint latent space")
        constraint_latent_space.setBounds(-self.arena_size, self.arena_size)

        self.latent_space = ob.CompoundStateSpace()
        self.latent_space.addSubspace(dynamics_latent_space, 1)
        self.latent_space.addSubspace(constraint_latent_space, 0)

        self.control_space = oc.RealVectorControlSpace(self.latent_space, self.L)

        self.ss = oc.SimpleSetup(self.control_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(self.dt)
        self.si.setMinMaxControlDuration(1, 10)

        # use the default state propagator, it won't be called anyways because we use a directed control sampler
        self.ss.setStatePropagator(oc.StatePropagator(self.si))

        self.si.setDirectedControlSamplerAllocator(self.directed_control_sampler.allocator(self.action_selector))
        self.directed_control_sampler = self.si.allocDirectedControlSampler()

        self.planner = oc.RRT(self.si)
        self.planner.setIntermediateStates(False)
        self.ss.setPlanner(self.planner)

    def isStateValid(self, state):
        # check if our model predicts that constraints are violated in this state
        constraint_state = np.ndarray(self.P)
        for i in range(self.P):
            constraint_state[i] = state[1][i]
        constraint_violated = self.tf_model.constraint_violated(constraint_state)
        constraint_violated = np.any(constraint_violated)
        valid = self.latent_space.satisfiesBounds(state) and not constraint_violated
        return valid

    def act(self, sdf, o_d_start, o_k_start, o_d_goal, verbose=False):
        # create start and goal states
        self.directed_control_sampler.reset()
        start = ob.State(self.latent_space)
        goal = ob.State(self.latent_space)
        for i in range(self.M):
            start()[0][i] = o_d_start[i, 0].astype(np.float64)
            goal()[0][i] = o_d_goal[i, 0].astype(np.float64)
        for i in range(self.P):
            start()[1][i] = o_k_start[i, 0].astype(np.float64)
            goal()[1][i] = -999

        self.ss.clear()
        # the threshold on "cost-to-goal" is interpretable here as Euclidian distance
        epsilon = 0.1
        self.ss.setStartAndGoalStates(start, goal, 0.4 * epsilon)
        try:
            solved = self.ss.solve(self.planner_timeout)

            if verbose:
                print("Planning time: {}".format(self.ss.getLastPlanComputationTime()))

            if solved:
                ompl_path = self.ss.getSolutionPath()

                numpy_controls = np.ndarray((ompl_path.getControlCount(), 1, self.L))
                numpy_d_states = np.ndarray((ompl_path.getStateCount(), self.M, 1))
                numpy_k_states = np.ndarray((ompl_path.getStateCount(), self.P, 1))
                for i, state in enumerate(ompl_path.getStates()):
                    for j in range(self.M):
                        numpy_d_states[i, j] = state[0][j]
                    for j in range(self.P):
                        numpy_k_states[i, j] = state[1][j]
                for i, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
                    numpy_controls[i, 0, 0] = control[0]
                    numpy_controls[i, 0, 1] = control[1]

                # Verification
                # verified = self.verify(numpy_controls, numpy_d_states, numpy_k_states)
                # if not verified:
                #     print("ERROR! NOT VERIFIED!")

                # SMOOTHING
                numpy_d_states, numpy_k_states, numpy_controls = self.smooth(numpy_d_states, numpy_k_states,
                                                                             numpy_controls,
                                                                             verbose)

                if verbose:
                    self.directed_control_sampler.plot_dual_sdf(sdf, o_d_start, o_d_goal, numpy_d_states,
                                                                numpy_k_states, numpy_controls, self.arena_size)
                    plt.show()
                    final_error = np.linalg.norm(numpy_d_states[-1] - o_d_goal)
                    lengths = [np.linalg.norm(numpy_d_states[i] - numpy_d_states[i - 1]) for i in
                               range(1, len(numpy_d_states))]
                    path_length = np.sum(lengths)
                    duration = self.dt * len(numpy_d_states)
                    print("Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s".format(
                        final_error, path_length, len(numpy_d_states), duration))
                return numpy_controls, numpy_d_states
            else:
                raise RuntimeError("No Solution found from {} to {}".format(start, goal))
        except RuntimeError:
            return None, None

    def smooth(self, numpy_d_states, numpy_k_states, numpy_controls, iters=50, verbose=False):
        new_d_states = list(numpy_d_states)
        new_k_states = list(numpy_k_states)
        new_controls = list(numpy_controls)
        shortcut_iter = 0
        shortcut_successes = 0
        while shortcut_iter < iters:
            shortcut_iter += 1
            # bias starting towards the beginning?
            start_idx = np.random.randint(0, len(new_d_states) / 2)
            # start_idx = np.random.randint(0, len(new_d_states) - 1)
            d_shortcut_start = new_d_states[start_idx]
            k_shortcut_start = new_k_states[start_idx]
            end_idx = np.random.randint(start_idx + 1, len(new_d_states))
            d_shortcut_end = new_d_states[end_idx]

            success, new_shortcut_us, new_shortcut_o_ds, new_shortcut_o_ks = self.directed_control_sampler.dual_shortcut(
                d_shortcut_start, k_shortcut_start, d_shortcut_end)

            if success:
                # popping changes the indexes of everything, so we just pop at start_idx the right number of times
                for i in range(start_idx, end_idx):
                    new_d_states.pop(start_idx)
                    new_k_states.pop(start_idx)
                    new_controls.pop(start_idx)
                for i, (shortcut_u, shortcut_o_d, shortcut_o_k) in enumerate(
                        zip(new_shortcut_us, new_shortcut_o_ds, new_shortcut_o_ks)):
                    new_d_states.insert(start_idx + i, shortcut_o_d)  # or maybe shortcut_end?
                    new_k_states.insert(start_idx + i, shortcut_o_k)  # or maybe shortcut_end?
                    new_controls.insert(start_idx + i, shortcut_u.reshape(1, 2))
                shortcut_successes += 1

        if verbose:
            print("{}/{} shortcuts succeeded".format(shortcut_successes, shortcut_iter))

        numpy_d_states = np.array(new_d_states)
        numpy_k_states = np.array(new_k_states)
        numpy_controls = np.array(new_controls)

        return numpy_d_states, numpy_k_states, numpy_controls

    def verify(self, controls, o_ds, o_ks):
        o_d = o_ds[0]
        o_k = o_ks[0]
        for i, u in enumerate(controls):

            if not np.allclose(o_d, o_ds[i]):
                return False
            if not np.allclose(o_k, o_ks[i]):
                return False
            constraint_violated = self.tf_model.constraint_violated(o_k.squeeze())
            if constraint_violated:
                return False

            o_d_next, o_k_next = self.tf_model.simple_dual_predict(o_d, o_k, u.reshape(2, 1))

            o_d = o_d_next
            o_k = o_k_next

        return True
