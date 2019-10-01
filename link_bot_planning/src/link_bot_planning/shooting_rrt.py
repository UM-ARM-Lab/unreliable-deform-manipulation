import numpy as np
from matplotlib import animation
from ompl import base as ob
from ompl import control as oc

from link_bot_planning.link_bot_goal import LinkBotGoal
from link_bot_planning.ompl_util import plot
from link_bot_planning.shooting_directed_control_sampler import ShootingDirectedControlSampler
from link_bot_planning.state_spaces import LinkBotControlSpace, TailStateSpaceSampler, from_numpy, to_numpy


class ShootingRRT:

    def __init__(self, fwd_model, constraint_checker_wrapper, dt, max_v, planner_timeout, n_state, env_w, env_h):
        self.constraint_checker_wrapper = constraint_checker_wrapper
        self.fwd_model = fwd_model
        self.dt = dt
        self.planner_timeout = planner_timeout
        self.n_state = self.fwd_model.n_state
        self.n_control = self.fwd_model.n_control
        self.max_v = max_v

        self.extent = [-env_w / 2, env_w / 2, -env_h / 2, env_h / 2]
        self.state_space = ob.RealVectorStateSpace(n_state)
        self.state_space.setName("rope configuration space")
        bounds = ob.RealVectorBounds(self.n_state)
        bounds.setLow(0, -env_w / 2)
        bounds.setLow(1, -env_h / 2)
        bounds.setLow(2, -env_w / 2)
        bounds.setLow(3, -env_h / 2)
        bounds.setLow(4, -env_w / 2)
        bounds.setLow(5, -env_h / 2)
        bounds.setHigh(0, env_w / 2)
        bounds.setHigh(1, env_h / 2)
        bounds.setHigh(2, env_w / 2)
        bounds.setHigh(3, env_h / 2)
        bounds.setHigh(4, env_w / 2)
        bounds.setHigh(5, env_h / 2)
        self.state_space.setBounds(bounds)

        def state_sampler_allocator(state_space):
            sampler = TailStateSpaceSampler(state_space, self.extent)
            return sampler

        self.state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler_allocator))

        self.control_space = LinkBotControlSpace(self.state_space, self.n_control)
        control_bounds = ob.RealVectorBounds(2)
        control_bounds.setLow(-max_v)
        control_bounds.setHigh(max_v)
        self.control_space.setBounds(control_bounds)

        self.ss = oc.SimpleSetup(self.control_space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(self.dt)
        self.si.setMinMaxControlDuration(1, 100)

        # use the default state propagator, it won't be called anyways because we use a directed control sampler
        # self.ss.setStatePropagator(oc.StatePropagator(self.si))
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))

        self.si.setDirectedControlSamplerAllocator(
            ShootingDirectedControlSampler.allocator(self.fwd_model, self.max_v))

        self.planner = oc.RRT(self.si)
        self.planner.setIntermediateStates(False)
        self.ss.setPlanner(self.planner)

        self.Writer = animation.writers['ffmpeg']
        self.writer = self.Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    def propagate(self, start, control, duration, state_out):
        np_s = to_numpy(start, self.n_state)
        np_u = to_numpy(control, self.n_control)

        np_s_next = self.fwd_model.fwd_act(np_s, np_u)

        from_numpy(np_s_next, state_out, self.n_state)

    def isStateValid(self, state):
        with self.constraint_checker_wrapper.get_graph().as_default():
            # check if our model predicts that constraints are violated in this state
            if not self.state_space.satisfiesBounds(state):
                return False

            np_state = to_numpy(state, self.n_state)
            p_reject = self.constraint_checker_wrapper(np_state)
            reject = np.random.uniform(0, 1) < p_reject
            return not reject

    def plan(self, np_start, tail_goal_point, sdf, verbose=0):
        """
        :param np_start: 1 by n matrix
        :param tail_goal_point:  1 by n matrix
        :param sdf:
        :param verbose: an integer, lower means less verbose
        :return:
        """
        # create start and goal states
        start = ob.State(self.state_space)
        from_numpy(np_start, start, self.n_state)
        epsilon = 0.10
        goal = LinkBotGoal(self.si, epsilon, tail_goal_point)

        self.ss.clear()
        self.ss.setStartState(start)
        self.ss.setGoal(goal)
        try:
            solved = self.ss.solve(self.planner_timeout)
            planning_time = self.ss.getLastPlanComputationTime()

            if verbose >= 2:
                print("Planning time: {}".format(planning_time))

            if solved:
                ompl_path = self.ss.getSolutionPath()

                np_states = np.ndarray((ompl_path.getStateCount(), self.n_state))
                np_controls = np.ndarray((ompl_path.getControlCount(), self.n_control))
                for i, state in enumerate(ompl_path.getStates()):
                    np_states[i] = to_numpy(state, self.n_state)
                for i, control in enumerate(ompl_path.getControls()):
                    np_controls[i] = to_numpy(control, self.n_control)

                # Verification
                # verified = self.verify(np_controls, np_states)
                # if not verified:
                #     print("ERROR! NOT VERIFIED!")

                # SMOOTHING
                # np_states, np_controls = self.smooth(np_states, np_controls, verbose)

                if verbose >= 2:
                    planner_data = ob.PlannerData(self.si)
                    self.planner.getPlannerData(planner_data)
                    plot(planner_data, sdf, np_start, tail_goal_point, np_states, np_controls, self.n_state, self.extent)
                    final_error = np.linalg.norm(np_states[-1, 0:2] - tail_goal_point)
                    lengths = [np.linalg.norm(np_states[i] - np_states[i - 1]) for i in range(1, len(np_states))]
                    path_length = np.sum(lengths)
                    duration = self.dt * len(np_states)
                    print("Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s".format(
                        final_error, path_length, len(np_states), duration))

                return np_controls, np_states, planning_time
            else:
                raise RuntimeError("No Solution found from {} to {}".format(start, goal))
        except RuntimeError:
            return None, None, -1

    def smooth(self, np_states, np_controls, iters=50, verbose=0):
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

            success, new_shortcut_us, new_shortcut_ss, = ShootingDirectedControlSampler.shortcut(d_shortcut_start, d_shortcut_end)

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

        if verbose >= 3:
            print("{}/{} shortcuts succeeded".format(shortcut_successes, shortcut_iter))

        np_states = np.array(new_states)
        np_controls = np.array(new_controls)

        return np_states, np_controls

    def verify(self, controls, ss):
        s = ss[0]
        for i, u in enumerate(controls):

            if not np.allclose(s, ss[i]):
                return False
            constraint_violated = self.fwd_model.constraint_violated(s.squeeze())
            if constraint_violated:
                return False

            s_next = self.fwd_model.simple_dual_predict(s, u.reshape(2, 1))

            s = s_next

        return True
