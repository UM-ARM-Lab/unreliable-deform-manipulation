from typing import Tuple, List

import numpy as np
from ompl import base as ob
from ompl import control as oc

from link_bot_gazebo.gazebo_utils import GazeboServices, get_local_occupancy_data
from link_bot_planning.link_bot_goal import LinkBotCompoundGoal
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.viz_object import VizObject
from link_bot_planning.params import EnvParams, PlannerParams
from link_bot_planning.shooting_directed_control_sampler import ShootingDirectedControlSampler
from link_bot_planning.state_spaces import to_numpy, ValidRopeConfigurationCompoundSampler, to_numpy_local_env, from_numpy
from link_bot_pycommon import link_bot_sdf_utils


class ShootingRRT(MyPlanner):

    def __init__(self,
                 fwd_model,
                 classifier_model,
                 planner_params: PlannerParams,
                 env_params: EnvParams,
                 services: GazeboServices,
                 viz_object: VizObject):
        super().__init__(fwd_model,
                         classifier_model,
                         planner_params,
                         env_params,
                         services,
                         viz_object)
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.n_state = self.fwd_model.n_state
        self.n_control = self.fwd_model.n_control
        self.env_params = env_params
        self.planner_params = planner_params
        self.services = services
        self.viz_object = viz_object

        self.state_space = ob.CompoundStateSpace()
        self.n_local_env = self.fwd_model.local_env_params.w_cols * self.fwd_model.local_env_params.h_rows
        self.local_env_space = ob.RealVectorStateSpace(self.n_local_env)
        epsilon = 1e-3
        self.local_env_space.setBounds(-epsilon, 1 + epsilon)

        self.local_env_origin_space = ob.RealVectorStateSpace(2)
        self.local_env_origin_space.setBounds(-10000, 10000)

        self.config_space = ob.RealVectorStateSpace(self.n_state)
        self.config_space_bounds = ob.RealVectorBounds(self.n_state)
        self.config_space_bounds.setLow(0, -self.env_params.w / 2)
        self.config_space_bounds.setLow(1, -self.env_params.h / 2)
        self.config_space_bounds.setLow(2, -self.env_params.w / 2)
        self.config_space_bounds.setLow(3, -self.env_params.h / 2)
        self.config_space_bounds.setLow(4, -self.env_params.w / 2)
        self.config_space_bounds.setLow(5, -self.env_params.h / 2)
        self.config_space_bounds.setHigh(0, self.env_params.w / 2)
        self.config_space_bounds.setHigh(1, self.env_params.h / 2)
        self.config_space_bounds.setHigh(2, self.env_params.w / 2)
        self.config_space_bounds.setHigh(3, self.env_params.h / 2)
        self.config_space_bounds.setHigh(4, self.env_params.w / 2)
        self.config_space_bounds.setHigh(5, self.env_params.h / 2)
        self.config_space.setBounds(self.config_space_bounds)

        # the rope is just 6 real numbers with no bounds
        self.state_space.addSubspace(self.config_space, weight=1.0)
        # the local environment is a rows*cols flat vector of numbers from 0 to 1
        self.state_space.addSubspace(self.local_env_space, weight=0.0)
        # origin
        self.state_space.addSubspace(self.local_env_origin_space, weight=0.0)

        # Only sample configurations which are known to be valid, i.e. not overstretched.
        def state_sampler_allocator(state_space):
            # this length comes from the SDF file textured_link_bot.sdf
            sampler = ValidRopeConfigurationCompoundSampler(state_space, self.viz_object, extent=self.env_params.extent,
                                                            link_length=0.24)
            return sampler

        self.state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(state_sampler_allocator))

        control_bounds = ob.RealVectorBounds(2)
        control_bounds.setLow(-self.planner_params.max_v)
        control_bounds.setHigh(self.planner_params.max_v)
        self.control_space = oc.RealVectorControlSpace(self.state_space, self.n_control)
        self.control_space.setBounds(control_bounds)

        self.ss = oc.SimpleSetup(self.control_space)

        self.si = self.ss.getSpaceInformation()

        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))

        self.planner = oc.RRT(self.si)
        self.planner.setIntermediateStates(True)  # this is necessary!
        self.ss.setPlanner(self.planner)
        self.si.setPropagationStepSize(self.fwd_model.dt)
        self.si.setMinMaxControlDuration(1, 50)

    def is_valid(self, state):
        return self.state_space.getSubspace(0).satisfiesBounds(state[0])

    def propagate(self, start, control, duration, state_out):
        del duration  # unused, mult-step propogation is handled inside propagateWhileValid
        np_s = to_numpy(start[0], self.n_state)
        np_u = np.expand_dims(to_numpy(control, self.n_control), axis=0)
        local_env_data = self.get_local_env_at(np_s[0, 4], np_s[0, 5])

        # use the forward model to predict the next configuration
        points_next = self.fwd_model.predict(local_env_data=[local_env_data], first_states=np_s, actions=np_u)
        np_s_next = points_next[:, 1].reshape([1, self.n_state])

        # validate the edge
        accept_probability = self.classifier_model.predict(local_env_data, np_s, np_s_next)
        edge_is_valid = np.random.uniform(0, 1) <= accept_probability

        # copy the result into the ompl state data structure
        if not edge_is_valid:
            # This will ensure this edge is not added to the tree
            self.viz_object.rejected_samples.append(np_s_next[0])
            state_out[0][0] = 1000
        else:
            from_numpy(np_s_next, state_out[0], self.n_state)
            local_env_data = self.get_local_env_at(np_s_next[0, 4], np_s_next[0, 5])
            local_env = local_env_data.data.flatten().astype(np.float64)
            for idx in range(self.n_local_env):
                occupancy_value = local_env[idx]
                state_out[1][idx] = occupancy_value
            origin = local_env_data.origin.astype(np.float64)
            from_numpy(origin, state_out[2], 2)

    def plan(self, np_start: np.ndarray,
             tail_goal_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[link_bot_sdf_utils.OccupancyData]]:
        """
        :param np_start: 1 by n matrix
        :param tail_goal_point:  1 by n matrix
        :return: controls, states
        """
        # create start and goal states
        start_local_occupancy = self.get_local_env_at(np_start[0, 4], np_start[0, 5])
        compound_start = ob.CompoundState(self.state_space)
        for i in range(self.n_state):
            compound_start()[0][i] = np_start[0, i]
        start_local_occupancy_flat_double = start_local_occupancy.data.flatten().astype(np.float64)
        for idx in range(self.n_local_env):
            occupancy_value = start_local_occupancy_flat_double[idx]
            compound_start()[1][idx] = occupancy_value
        start_local_occupancy_origin_double = start_local_occupancy.origin.astype(np.float64)
        compound_start()[2][0] = start_local_occupancy_origin_double[0]
        compound_start()[2][1] = start_local_occupancy_origin_double[1]

        start = ob.State(compound_start)
        goal = LinkBotCompoundGoal(self.si, self.planner_params.goal_threshold, tail_goal_point)

        self.ss.clear()
        self.viz_object.clear()
        self.ss.setStartState(start)
        self.ss.setGoal(goal)
        solved = self.ss.solve(self.planner_params.timeout)

        if solved:
            ompl_path = self.ss.getSolutionPath()

            np_states = np.ndarray((ompl_path.getStateCount(), self.n_state))
            np_controls = np.ndarray((ompl_path.getControlCount(), self.n_control))
            planner_local_envs = []
            for i, state in enumerate(ompl_path.getStates()):
                np_s = to_numpy(state[0], self.n_state)
                np_states[i] = np_s
                grid = to_numpy_local_env(state[1], self.fwd_model.local_env_params.h_rows,
                                          self.fwd_model.local_env_params.w_cols)
                res_2d = np.array([self.fwd_model.local_env_params.res, self.fwd_model.local_env_params.res])
                origin = to_numpy(state[2], 2)[0]
                planner_local_env = link_bot_sdf_utils.OccupancyData(grid, res_2d, origin)
                planner_local_envs.append(planner_local_env)
            for i, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
                # duration is always be 1 for control::RRT, not so for control::SST
                np_controls[i] = to_numpy(control, self.n_control)

            return np_controls, np_states, planner_local_envs

        raise RuntimeError("No Solution found from {} to {}".format(start, goal))

    def get_local_env_at(self, x: float, y: float):
        center_point = np.array([x, y])
        return get_local_occupancy_data(rows=self.fwd_model.local_env_params.h_rows,
                                        cols=self.fwd_model.local_env_params.w_cols,
                                        res=self.fwd_model.local_env_params.res,
                                        center_point=center_point,
                                        services=self.services)

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
