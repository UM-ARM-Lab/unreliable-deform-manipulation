from ompl import base as ob
from ompl import control as oc

from time import time
import matplotlib.pyplot as plt
import numpy as np
from link_bot_agent.gp_directed_control_sampler import GPDirectedControlSampler, plot
from link_bot_agent.link_bot_goal import LinkBotGoal
from link_bot_agent import ompl_util
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
        np_state = np.ndarray((1, self.n_state))
        for i in range(self.n_state):
            np_state[0, i] = state[i]
        constraint_violated = self.constraint_violated(np_state)
        valid = self.state_space.satisfiesBounds(state) and not constraint_violated
        return valid

    def plan(self, np_start, np_goal, sdf, verbose=False):
        # create start and goal states
        start = ob.State(self.state_space)
        for i in range(self.n_state):
            start()[i] = np_start[0, i].astype(np.float64)
        # the threshold on "cost-to-goal" is interpretable here as Euclidean distance
        epsilon = 0.1
        goal = LinkBotGoal(self.si, epsilon, np_goal)

        self.ss.clear()
        self.ss.setStartState(start)
        self.ss.setGoal(goal)
        try:
            solved = self.ss.solve(self.planner_timeout)

            if verbose:
                print("Planning time: {}".format(self.ss.getLastPlanComputationTime()))

            if solved:
                ompl_path = self.ss.getSolutionPath()

                np_states = np.ndarray((ompl_path.getStateCount(), self.n_state))
                np_controls = np.ndarray((ompl_path.getControlCount(), self.n_control))
                np_durations = np.ndarray(ompl_path.getControlCount())
                for i, state in enumerate(ompl_path.getStates()):
                    for j in range(self.n_state):
                        np_states[i, j] = state[j]
                for i, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
                    np_durations[i] = duration
                    for j in range(self.n_control):
                        np_controls[i, j] = control[j]
                np_duration_steps_int = (np_durations / self.dt).astype(np.int)

                # Verification
                # verified = self.verify(np_controls, np_states)
                # if not verified:
                #     print("ERROR! NOT VERIFIED!")

                # SMOOTHING
                # np_states, np_controls = self.smooth(np_states, np_controls, verbose)

                if verbose:
                    planner_data = ob.PlannerData(self.si)
                    self.planner.getPlannerData(planner_data)
                    plot(planner_data, sdf, np_start, np_goal, np_states, np_controls, self.arena_size)
                    particles = link_bot_gp.predict(self.fwd_gp_model, np_start, np_controls, np_duration_steps_int)
                    animation = link_bot_gp.animate_predict(particles, sdf, self.arena_size)
                    animation.save('gp_mpc_{}.gif'.format(int(time())), writer='imagemagick', fps=10)
                    plt.show()
                    final_error = np.linalg.norm(np_states[-1, 0:2] - np_goal)
                    lengths = [np.linalg.norm(np_states[i] - np_states[i - 1]) for i in range(1, len(np_states))]
                    path_length = np.sum(lengths)
                    duration = self.dt * len(np_states)
                    print("Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s".format(
                        final_error, path_length, len(np_states), duration))
                np_controls_flat = ompl_util.flatten_plan(np_controls, np_duration_steps_int)
                return np_controls_flat, np_states
            else:
                raise RuntimeError("No Solution found from {} to {}".format(start, goal))
        except RuntimeError:
            return None, None

    def smooth(self, np_states, np_controls, iters=50, verbose=False):
        new_states = list(np_states)
        new_controls = list(np_controls)
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

        np_states = np.array(new_states)
        np_controls = np.array(new_controls)

        return np_states, np_controls

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
