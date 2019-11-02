from typing import Tuple, List

import numpy as np
from ompl import base as ob
from ompl import control as oc

from link_bot_gazebo.gazebo_utils import GazeboServices, get_local_occupancy_data
from link_bot_planning.link_bot_goal import LinkBotCompoundGoal
from link_bot_planning.mpc_planners import MyPlanner
from link_bot_planning.ompl_viz import VizObject
from link_bot_planning.params import EnvParams, LocalEnvParams, PlannerParams
from link_bot_planning.state_spaces import to_numpy, ValidRopeConfigurationCompoundSampler, to_numpy_local_env, from_numpy
from link_bot_pycommon import link_bot_sdf_utils


class SST(MyPlanner):

    def __init__(self,
                 fwd_model,
                 classifier_model,
                 dt: float,
                 planner_params: PlannerParams,
                 local_env_params: LocalEnvParams,
                 env_params: EnvParams,
                 services: GazeboServices,
                 viz_object: VizObject):
        super().__init__(fwd_model,
                         classifier_model,
                         dt,
                         planner_params,
                         local_env_params,
                         env_params,
                         services,
                         viz_object)

        self.state_space = ob.CompoundStateSpace()
        self.n_local_env = self.local_env_params.w_cols * self.local_env_params.h_rows
        self.local_env_space = ob.RealVectorStateSpace(self.n_local_env)
        self.local_env_space.setBounds(0, 10)

        self.local_env_origin_space = ob.RealVectorStateSpace(2)
        self.local_env_origin_space.setBounds(-10000.1, 10000.0)

        self.config_space = ob.RealVectorStateSpace(self.n_state)
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
        self.state_space.addSubspace(self.local_env_space, weight=0.0)
        # origin
        self.state_space.addSubspace(self.local_env_origin_space, weight=0.0)

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

        # SST will use this to propagate things
        self.ss.setStatePropagator(oc.StatePropagator(self.si))

        self.planner = oc.SST(self.si)
        print(self.planner.getRange())
        # self.planner.setRange(1.0)
        # self.planner.setPruningRadius(1.0)
        self.ss.setPlanner(self.planner)


def propagate(self, start, control, duration, state_out):
    print(duration)
    np_s = to_numpy(start, self.n_state)
    np_u = to_numpy(control, self.n_control)

    # use the forward model to predict the next configuration
    points_next = self.fwd_model.predict(np_s, np_u)
    np_s_next = points_next[:, 1].reshape([1, self.n_state])

    from_numpy(np_s_next, state_out, self.n_state)


def plan(self, np_start: np.ndarray,
         tail_goal_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[link_bot_sdf_utils.OccupancyData]]:
    """
    :param np_start: 1 by n matrix
    :param tail_goal_point:  1 by n matrix
    :return: controls, states
    """
    # create start and goal states
    start_local_occupancy = get_local_occupancy_data(rows=self.local_env_params.h_rows,
                                                     cols=self.local_env_params.w_cols,
                                                     res=self.local_env_params.res,
                                                     center_point=np.array([np_start[0, 4], np_start[0, 5]]),
                                                     services=self.services)
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
    epsilon = 0.01
    goal = LinkBotCompoundGoal(self.si, epsilon, tail_goal_point)

    self.ss.clear()
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
            grid = to_numpy_local_env(state[1], self.local_env_params.h_rows, self.local_env_params.w_cols)
            res_2d = np.array([self.local_env_params.res, self.local_env_params.res])
            origin = to_numpy(state[2], 2)[0]
            planner_local_env = link_bot_sdf_utils.OccupancyData(grid, res_2d, origin)
            planner_local_envs.append(planner_local_env)
        for i, control in enumerate(ompl_path.getControls()):
            np_controls[i] = to_numpy(control, self.n_control)

        return np_controls, np_states, planner_local_envs

    raise RuntimeError("No Solution found from {} to {}".format(start, goal))
