from ompl import base as ob
from ompl import control as oc

import matplotlib.pyplot as plt
import numpy as np
from link_bot_agent.gp_directed_control_sampler import GPDirectedControlSampler
from link_bot_agent.link_bot_goal import LinkBotGoal
from link_bot_gaussian_process import link_bot_gp


class GPRRT:

    def __init__(self, fwd_gp_model, inv_gp_model, constraint_violated, dt, max_v, planner_timeout):
        self.fwd_gp_model = fwd_gp_model
        self.inv_gp_model = inv_gp_model
        self.constraint_violated = constraint_violated
        self.dt = dt
        self.planner_timeout = planner_timeout
        self.n_state = self.fwd_gp_model.n_state
        self.n_control = self.fwd_gp_model.n_control
        self.max_v = max_v

        self.arena_size = 5
        self.state_space_size = 5
        self.state_space = ob.RealVectorStateSpace(self.n_state)
        self.state_space.setName("dynamics latent space")
        self.state_space.setBounds(-self.state_space_size, self.state_space_size)

        self.control_space = oc.RealVectorControlSpace(self.state_space, self.n_control)

        self.ss = oc.SimpleSetup(self.control_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(self.dt)
        self.si.setMinMaxControlDuration(1, 10)

        # use the default state propagator, it won't be called anyways because we use a directed control sampler
        self.ss.setStatePropagator(oc.StatePropagator(self.si))

        self.si.setDirectedControlSamplerAllocator(
            GPDirectedControlSampler.allocator(self.fwd_gp_model, self.inv_gp_model, self.max_v))

        self.planner = oc.RRT(self.si)
        self.planner.setIntermediateStates(False)
        self.ss.setPlanner(self.planner)

    def isStateValid(self, state):
        # check if our model predicts that constraints are violated in this state
        numpy_state = np.ndarray((self.n_state, 1))
        for i in range(self.n_state):
            numpy_state[i, 0] = state[i]
        constraint_violated = self.constraint_violated(numpy_state)
        valid = self.state_space.satisfiesBounds(state) and not constraint_violated
        return valid

    def plan(self, numpy_start, numpy_goal, sdf, verbose=False):
        # create start and goal states
        GPDirectedControlSampler.reset()
        start = ob.State(self.state_space)
        for i in range(self.n_state):
            start()[i] = numpy_start[i, 0].astype(np.float64)
        # the threshold on "cost-to-goal" is interpretable here as Euclidean distance
        epsilon = 0.1
        goal = LinkBotGoal(self.si, epsilon, numpy_goal)

        self.ss.clear()
        self.ss.setStartState(start)
        self.ss.setGoal(goal)
        try:
            solved = self.ss.solve(self.planner_timeout)

            if verbose:
                print("Planning time: {}".format(self.ss.getLastPlanComputationTime()))

            if solved:
                ompl_path = self.ss.getSolutionPath()

                numpy_states = np.ndarray((ompl_path.getStateCount(), self.n_state))
                numpy_controls = np.ndarray((ompl_path.getControlCount(), self.n_control))
                for i, state in enumerate(ompl_path.getStates()):
                    for j in range(self.n_state):
                        numpy_states[i, j] = state[j]
                for i, control in enumerate(ompl_path.getControls()):
                    for j in range(self.n_control):
                        numpy_controls[i, j] = control[j]

                # Verification
                # verified = self.verify(numpy_controls, numpy_states)
                # if not verified:
                #     print("ERROR! NOT VERIFIED!")

                # SMOOTHING
                # numpy_states, numpy_controls = self.smooth(numpy_states, numpy_controls, verbose)

                if verbose:
                    GPDirectedControlSampler.plot(sdf, numpy_start, numpy_goal, numpy_states, numpy_controls,
                                                  self.arena_size)
                    particles = link_bot_gp.predict(self.fwd_gp_model, numpy_start.T, numpy_controls)
                    animation = link_bot_gp.animate_predict(particles)
                    plt.show()
                    final_error = np.linalg.norm(numpy_states[-1, 0:2] - numpy_goal)
                    lengths = [np.linalg.norm(numpy_states[i] - numpy_states[i - 1]) for i in
                               range(1, len(numpy_states))]
                    path_length = np.sum(lengths)
                    duration = self.dt * len(numpy_states)
                    print("Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s".format(
                        final_error, path_length, len(numpy_states), duration))
                return numpy_controls, numpy_states
            else:
                raise RuntimeError("No Solution found from {} to {}".format(start, goal))
        except RuntimeError:
            return None, None

    def smooth(self, numpy_states, numpy_controls, iters=50, verbose=False):
        new_states = list(numpy_states)
        new_controls = list(numpy_controls)
        shortcut_iter = 0
        shortcut_successes = 0
        while shortcut_iter < iters:
            shortcut_iter += 1
            # bias starting towards the beginning?
            start_idx = np.random.randint(0, len(new_states) / 2)
            # start_idx = np.random.randint(0, len(new_states) - 1)
            d_shortcut_start = new_states[start_idx]
            end_idx = np.random.randint(start_idx + 1, len(new_states))
            d_shortcut_end = new_states[end_idx]

            success, new_shortcut_us, new_shortcut_ss, = GPDirectedControlSampler.shortcut(
                d_shortcut_start, d_shortcut_end)

            if success:
                # popping changes the indexes of everything, so we just pop at start_idx the right number of times
                for i in range(start_idx, end_idx):
                    new_states.pop(start_idx)
                    new_controls.pop(start_idx)
                for i, (shortcut_u, shortcut_s) in enumerate(
                        zip(new_shortcut_us, new_shortcut_ss)):
                    new_states.insert(start_idx + i, shortcut_s)  # or maybe shortcut_end?
                    new_controls.insert(start_idx + i, shortcut_u.reshape(1, 2))
                shortcut_successes += 1

        if verbose:
            print("{}/{} shortcuts succeeded".format(shortcut_successes, shortcut_iter))

        numpy_states = np.array(new_states)
        numpy_controls = np.array(new_controls)

        return numpy_states, numpy_controls

    def verify(self, controls, ss):
        s = ss[0]
        for i, u in enumerate(controls):

            if not np.allclose(s, ss[i]):
                return False
            constraint_violated = self.fwd_gp_model.constraint_violated(s.squeeze())
            if constraint_violated:
                return False

            s_next = self.fwd_gp_model.simple_dual_predict(s, u.reshape(2, 1))

            s = s_next

        return True
