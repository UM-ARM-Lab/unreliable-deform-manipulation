from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from ompl import base as ob
from ompl import control as oc

from link_bot_agent import ompl_util
from link_bot_agent.gp_directed_control_sampler import GPDirectedControlSampler, plot
from link_bot_agent.link_bot_goal import LinkBotGoal
from link_bot_gaussian_process import link_bot_gp


class LinkBotControlSpace(oc.RealVectorControlSpace):

    def __init__(self, state_space, n_control):
        super(LinkBotControlSpace, self).__init__(state_space, n_control)

    @staticmethod
    def to_numpy(control):
        assert isinstance(control, oc.RealVectorControlSpace.ControlType)
        np_u = np.ndarray((1, 2))
        np_u[0, 0] = control[0]
        np_u[0, 1] = control[1]
        return np_u


class MinimalLinkBotStateSpace(ob.RealVectorStateSpace):

    def __init__(self, n_state):
        super(MinimalLinkBotStateSpace, self).__init__(n_state)
        self.setDimensionName(0, 'tail_x')
        self.setDimensionName(1, 'tail_y')
        self.setDimensionName(2, 'theta_0')
        self.setDimensionName(3, 'theta_1')

    @staticmethod
    def to_numpy(state, l=0.5):
        np_s = np.ndarray((1, 6))
        np_s[0, 0] = state[0]
        np_s[0, 1] = state[1]
        np_s[0, 2] = np_s[0, 0] + np.cos(state[2]) * l
        np_s[0, 3] = np_s[0, 1] + np.sin(state[2]) * l
        np_s[0, 4] = np_s[0, 2] + np.cos(state[3]) * l
        np_s[0, 5] = np_s[0, 3] + np.sin(state[3]) * l
        return np_s

    @staticmethod
    def from_numpy(np_s, state_out):
        state_out[0] = np_s[0, 0]
        state_out[1] = np_s[0, 1]
        state_out[2] = np.arctan2(np_s[0, 3] - np_s[0, 1], np_s[0, 2] - np_s[0, 0])
        state_out[3] = np.arctan2(np_s[0, 5] - np_s[0, 3], np_s[0, 4] - np_s[0, 2])


class LinkBotStateSpace(ob.RealVectorStateSpace):

    def __init__(self, n_state):
        super(LinkBotStateSpace, self).__init__(n_state)
        self.setDimensionName(0, 'tail_x')
        self.setDimensionName(1, 'tail_y')
        self.setDimensionName(2, 'mid_x')
        self.setDimensionName(3, 'mid_y')
        self.setDimensionName(4, 'head_x')
        self.setDimensionName(5, 'head_y')

    #
    # def distance(self, s1, s2):
    #     # all the weight is in the tail
    #     weights = [1, 1, 0, 0, 0, 0]
    #     dist = 0
    #     for i in range(self.getDimension()):
    #         dist += weights[i] * (s1[i] - s2[i]) ** 2
    #     return dist

    @staticmethod
    def to_numpy(state):
        np_s = np.ndarray((1, 6))
        for i in range(6):
            np_s[0, i] = state[i]
        return np_s

    @staticmethod
    def from_numpy(np_s, state_out):
        for i in range(6):
            state_out[i] = np_s[0, i]


class GPRRT:

    def __init__(self, fwd_gp_model, inv_gp_model, constraint_checker_wrapper, dt, max_v, planner_timeout):
        self.constraint_checker_wrapper = constraint_checker_wrapper
        self.fwd_gp_model = fwd_gp_model
        self.inv_gp_model = inv_gp_model
        self.dt = dt
        self.planner_timeout = planner_timeout
        self.n_state = self.fwd_gp_model.n_state
        self.n_control = self.fwd_gp_model.n_control
        self.max_v = max_v

        self.arena_size = 5
        self.state_space_size = 5  # be careful this will cause out of bounds in the SDF
        self.state_space = LinkBotStateSpace(self.n_state)
        self.state_space.setName("dynamics latent space")
        self.state_space.setBounds(-self.state_space_size, self.state_space_size)

        self.control_space = LinkBotControlSpace(self.state_space, self.n_control)
        control_bounds = ob.RealVectorBounds(2)
        control_bounds.setLow(-0.3)
        control_bounds.setHigh(0.3)
        self.control_space.setBounds(control_bounds)

        self.ss = oc.SimpleSetup(self.control_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(self.dt)
        self.si.setMinMaxControlDuration(1, 100)

        # use the default state propagator, it won't be called anyways because we use a directed control sampler
        # self.ss.setStatePropagator(oc.StatePropagator(self.si))
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        # self.si.setDirectedControlSamplerAllocator(
        #     GPDirectedControlSampler.allocator(self.fwd_gp_model, self.inv_gp_model, self.max_v))

        self.planner = oc.RRT(self.si)
        self.planner.setIntermediateStates(False)
        self.ss.setPlanner(self.planner)

        self.Writer = animation.writers['ffmpeg']
        self.writer = self.Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    def propagate(self, start, control, duration, state_out):
        np_s = self.state_space.to_numpy(start)
        np_u = self.control_space.to_numpy(control)

        steps = int(duration / self.dt)
        np_s_next = np_s
        for t in range(steps):
            np_s_next = self.fwd_gp_model.fwd_act(np_s_next, np_u)

        # DEBUGGING:
        # sx = [np_s[0, 0], np_s[0, 2], np_s[0, 4]]
        # sy = [np_s[0, 1], np_s[0, 3], np_s[0, 5]]
        # sx_ = [np_s_next[0, 0], np_s_next[0, 2], np_s_next[0, 4]]
        # sy_ = [np_s_next[0, 1], np_s_next[0, 3], np_s_next[0, 5]]
        # sx_ = [np_s_next[0, 0], np_s_next[0, 2], np_s_next[0, 4]]
        # sy_ = [np_s_next[0, 1], np_s_next[0, 3], np_s_next[0, 5]]
        # plt.scatter(sx, sy, c=['b', 'b', 'g'], zorder=2)
        # plt.plot(sx, sy, c='k', zorder=1)
        # plt.scatter(sx_, sy_, c=['r', 'r', 'c'], zorder=2)
        # plt.quiver(np_s[0, 4], np_s[0, 5], control[0], control[1])
        # plt.axis("equal")
        # plt.show()

        self.state_space.from_numpy(np_s_next, state_out)

    def isStateValid(self, state):
        with self.constraint_checker_wrapper.get_graph().as_default():
            # check if our model predicts that constraints are violated in this state
            if not self.state_space.satisfiesBounds(state):
                return False

            np_state = self.state_space.to_numpy(state)
            constraint_violated = self.constraint_checker_wrapper(np_state)
            return not constraint_violated

    def plan(self, np_start, np_goal, sdf, verbose=False):
        # create start and goal states
        start = ob.State(self.state_space)
        self.state_space.from_numpy(np_start, start)
        epsilon = 0.10
        goal = LinkBotGoal(self.si, epsilon, np_goal)

        self.ss.clear()
        self.ss.setStartState(start)
        self.ss.setGoal(goal)
        try:
            solved = self.ss.solve(self.planner_timeout)
            planning_time = self.ss.getLastPlanComputationTime()

            if verbose:
                print("Planning time: {}".format(planning_time))

            if solved:
                ompl_path = self.ss.getSolutionPath()

                np_states = np.ndarray((ompl_path.getStateCount(), self.n_state))
                np_controls = np.ndarray((ompl_path.getControlCount(), self.n_control))
                np_durations = np.ndarray(ompl_path.getControlCount())
                for i, state in enumerate(ompl_path.getStates()):
                    np_states[i] = self.state_space.to_numpy(state)
                for i, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
                    np_durations[i] = duration
                    np_controls[i] = self.control_space.to_numpy(control)
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
                    plot(self.state_space, self.control_space, planner_data, sdf, np_start, np_goal, np_states, np_controls,
                         self.arena_size)
                    prediction, variances = link_bot_gp.predict(self.fwd_gp_model, np_start, np_controls, np_duration_steps_int)
                    animation = link_bot_gp.animate_predict(prediction, sdf, self.arena_size)
                    animation.save('gp_mpc_{}.mp4'.format(int(time())), writer=self.writer)
                    plt.show()
                    final_error = np.linalg.norm(np_states[-1, 0:2] - np_goal)
                    lengths = [np.linalg.norm(np_states[i] - np_states[i - 1]) for i in range(1, len(np_states))]
                    path_length = np.sum(lengths)
                    duration = self.dt * len(np_states)
                    print("Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s".format(
                        final_error, path_length, len(np_states), duration))
                np_controls_flat = ompl_util.flatten_plan(np_controls, np_duration_steps_int)
                return np_controls_flat, np_states, planning_time
            else:
                raise RuntimeError("No Solution found from {} to {}".format(start, goal))
        except RuntimeError:
            return None, None, -1

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
