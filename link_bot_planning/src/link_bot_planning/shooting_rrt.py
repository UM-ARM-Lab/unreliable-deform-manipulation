from typing import Tuple, List

import numpy as np
from ompl import base as ob
from ompl import control as oc

from link_bot_gazebo.gazebo_utils import GazeboServices, get_local_sdf_data
from link_bot_planning.link_bot_goal import LinkBotGoal
from link_bot_planning.params import EnvParams, SDFParams, PlannerParams
from link_bot_planning.shooting_directed_control_sampler import ShootingDirectedControlSampler
from link_bot_planning.state_spaces import to_numpy, \
    ValidRopeConfigurationCompoundSampler, from_numpy_compound
from link_bot_pycommon import link_bot_sdf_utils


class ShootingRRT:

    def __init__(self, fwd_model,
                 classifier_model,
                 dt: float,
                 n_state: int,
                 planner_params: PlannerParams,
                 sdf_params: SDFParams,
                 env_params: EnvParams,
                 services: GazeboServices):
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.dt = dt
        self.n_state = self.fwd_model.n_state
        self.n_control = self.fwd_model.n_control
        self.sdf_params = sdf_params
        self.env_params = env_params
        self.planner_params = planner_params
        self.services = services

        self.state_space = ob.CompoundStateSpace()
        self.n_local_sdf = self.sdf_params.local_w_cols * self.sdf_params.local_h_rows
        self.local_sdf_space = ob.RealVectorStateSpace(self.n_local_sdf)
        self.local_sdf_space.setBounds(-10, 10)

        self.config_space = ob.RealVectorStateSpace(n_state)
        bounds = ob.RealVectorBounds(self.n_state)
        bounds.setLow(0, -self.env_params.w / 2)
        bounds.setLow(1, -self.env_params.h / 2)
        bounds.setLow(2, -self.env_params.w / 2)
        bounds.setLow(3, -self.env_params.h / 2)
        bounds.setLow(4, -self.env_params.w / 2)
        bounds.setLow(5, -self.env_params.h / 2)
        bounds.setHigh(0, self.env_params.w / 2)
        bounds.setHigh(1, self.env_params.h / 2)
        bounds.setHigh(2, self.env_params.w / 2)
        bounds.setHigh(3, self.env_params.h / 2)
        bounds.setHigh(4, self.env_params.w / 2)
        bounds.setHigh(5, self.env_params.h / 2)
        self.config_space.setBounds(bounds)

        # the rope is just 6 real numbers with no bounds
        self.state_space.addSubspace(self.config_space, weight=1.0)
        # the local environment is a rows*cols flat vector of numbers from 0 to 1
        self.state_space.addSubspace(self.local_sdf_space, weight=0.0)

        # Only sample configurations which are known to be valid, i.e. not overstretched.
        def state_sampler_allocator(state_space):
            # this length comes from the SDF file textured_link_bot.sdf
            # sampler = ValidRopeConfigurationSampler(state_space, extent=self.env_params.extent, link_length=0.24)
            sampler = ValidRopeConfigurationCompoundSampler(state_space, extent=self.env_params.extent, link_length=0.24)
            return sampler

        self.state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler_allocator))

        control_bounds = ob.RealVectorBounds(2)
        control_bounds.setLow(-self.planner_params.max_v)
        control_bounds.setHigh(self.planner_params.max_v)
        self.control_space = oc.RealVectorControlSpace(self.state_space, self.n_control)
        self.control_space.setBounds(control_bounds)

        self.ss = oc.SimpleSetup(self.control_space)

        self.si = self.ss.getSpaceInformation()

        # use the default state propagator, it won't be called anyways because we use a directed control sampler
        self.ss.setStatePropagator(oc.StatePropagator(self.si))

        self.si.setDirectedControlSamplerAllocator(
            ShootingDirectedControlSampler.allocator(self.fwd_model,
                                                     self.classifier_model,
                                                     self.services,
                                                     self.sdf_params,
                                                     self.planner_params.max_v))

        self.planner = oc.RRT(self.si)
        self.planner.setIntermediateStates(False)
        self.ss.setPlanner(self.planner)

    def plan(self, np_start: np.ndarray,
             tail_goal_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[link_bot_sdf_utils.SDF]]:
        """
        :param np_start: 1 by n matrix
        :param tail_goal_point:  1 by n matrix
        :return: controls, states
        """
        # create start and goal states
        start_local_sdf = get_local_sdf_data(sdf_rows=self.sdf_params.local_h_rows,
                                             sdf_cols=self.sdf_params.local_w_cols,
                                             res=self.sdf_params.res,
                                             origin_point=np.array([np_start[0, 4], np_start[0, 5]]),
                                             services=self.services)
        compound_start = ob.CompoundState(self.state_space)
        for i in range(self.n_state):
            compound_start()[0][i] = np_start[0, i]
        compound_start()[1][0] = np.pi
        start_local_sdf_flat_double = start_local_sdf.sdf.flatten().astype(np.float64)
        for sdf_idx in range(self.n_local_sdf):
            sdf_value = start_local_sdf_flat_double[sdf_idx]
            compound_start()[1][sdf_idx] = sdf_value
        # from_numpy_compound(np_start, start_local_sdf, compound_start(), self.n_state)
        start = ob.State(compound_start)
        epsilon = 0.01
        goal = LinkBotGoal(self.si, epsilon, tail_goal_point)

        self.ss.clear()
        self.ss.setStartState(start)
        self.ss.setGoal(goal)
        solved = self.ss.solve(self.planner_params.timeout)

        if solved:
            ompl_path = self.ss.getSolutionPath()

            np_states = np.ndarray((ompl_path.getStateCount(), self.n_state))
            np_controls = np.ndarray((ompl_path.getControlCount(), self.n_control))
            planner_local_sdfs = []
            for i, state in enumerate(ompl_path.getStates()):
                np_states[i] = to_numpy(state[0], self.n_state)
            for i, control in enumerate(ompl_path.getControls()):
                np_controls[i] = to_numpy(control, self.n_control)

            # TODO: how to get the local SDFs out of the planner?

            # Verification
            # verified = self.verify(np_controls, np_states)
            # if not verified:
            #     print("ERROR! NOT VERIFIED!")

            # SMOOTHING
            # np_states, np_controls = self.smooth(np_states, np_controls, verbose)

            return np_controls, np_states, planner_local_sdfs

        raise RuntimeError("No Solution found from {} to {}".format(start, goal))

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
