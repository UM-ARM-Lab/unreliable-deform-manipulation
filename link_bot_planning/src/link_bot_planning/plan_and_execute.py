#!/usr/bin/env python
from __future__ import division, print_function

import time
from typing import Dict, Optional

import numpy as np
import std_srvs
from colorama import Fore
from link_bot_gazebo.srv import LinkBotStateRequest
from ompl import base as ob

import ignition.markers
from link_bot_gazebo import gazebo_services
from link_bot_planning import my_planner
from link_bot_planning.goals import sample_collision_free_goal
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import SimParams
from link_bot_pycommon import link_bot_sdf_utils, ros_pycommon
from link_bot_pycommon.ros_pycommon import Services, get_start_states
from link_bot_pycommon.ros_pycommon import get_occupancy_data


class PlanAndExecute:

    def __init__(self,
                 planner: MyPlanner,
                 n_total_plans: int,
                 n_plans_per_env: int,
                 verbose: int,
                 planner_params: Dict,
                 sim_params: SimParams,
                 services: Services,
                 no_execution: bool,
                 seed: int,
                 retry_on_failure: Optional[bool] = True,
                 pause_between_plans: Optional[bool] = False):
        self.pause_between_plans = pause_between_plans
        self.retry_on_failure = retry_on_failure
        self.planner = planner
        self.n_total_plans = n_total_plans
        self.n_plans_per_env = n_plans_per_env
        self.sim_params = sim_params
        self.planner_params = planner_params
        self.verbose = verbose
        self.services = services
        self.no_execution = no_execution
        self.env_rng = np.random.RandomState(seed)
        self.goal_rng = np.random.RandomState(seed)

        # remove all markers
        self.services.marker_provider.remove_all()

        self.plan_idx = 0
        self.total_plan_idx = 0

    def run(self):
        self.total_plan_idx = 0
        initial_poses_in_collision = 0
        while True:
            self.on_before_plan()

            # generate a bunch of plans to random goals
            state_req = LinkBotStateRequest()

            self.plan_idx = 0
            while True:
                # get full env once
                full_env_data = get_occupancy_data(env_w=self.planner.full_env_params.w,
                                                   env_h=self.planner.full_env_params.h,
                                                   res=self.planner.full_env_params.res,
                                                   services=self.services)

                # get start states
                start_states, link_bot_start_state, head_point = get_start_states(self.services,
                                                                                  self.planner.subspaces_to_plan_with.keys())

                # generate a random target
                tail_goal = np.array(self.get_goal(self.planner_params['random_goal_w'],
                                                   self.planner_params['random_goal_h'],
                                                   head_point,
                                                   env_padding=self.sim_params.goal_padding,
                                                   full_env_data=full_env_data))

                # plan to that target
                if self.verbose >= 1:
                    # tail start x,y and tail goal x,y
                    ignition.markers.publish_markers(self.services.marker_provider,
                                                     tail_goal[0], tail_goal[1],
                                                     link_bot_start_state[0],
                                                     link_bot_start_state[1],
                                                     marker_size=0.05)
                if self.verbose >= 1:
                    print(Fore.CYAN + "Planning from {} to {}".format(link_bot_start_state, tail_goal) + Fore.RESET)

                t0 = time.time()
                planner_results = self.planner.plan(start_states, tail_goal, full_env_data)
                planned_actions, planned_path_dict, planner_status = planner_results
                my_planner.interpret_planner_status(planner_status, self.verbose)
                planner_data = ob.PlannerData(self.planner.si)
                self.planner.planner.getPlannerData(planner_data)

                if self.verbose >= 1:
                    print(planner_status.asString())

                self.on_after_plan()

                if not planner_status:
                    self.on_planner_failure(link_bot_start_state, tail_goal, full_env_data, planner_data)
                    if self.retry_on_failure:
                        break
                else:  # Approximate or Exact solution found!
                    planning_time = time.time() - t0
                    if self.verbose >= 1:
                        print("Planning time: {:5.3f}s".format(planning_time))

                    self.on_plan_complete(planned_path_dict, tail_goal, planned_actions, full_env_data, planner_data,
                                          planning_time,
                                          planner_status)

                    trajectory_execution_request = ros_pycommon.make_trajectory_execution_request(self.planner.fwd_model.dt,
                                                                                                  planned_actions)

                    # execute the plan, collecting the states that actually occurred
                    if not self.no_execution:
                        if self.verbose >= 2:
                            print(Fore.CYAN + "Executing Plan.".format(tail_goal) + Fore.RESET)

                        traj_exec_response = self.services.execute_trajectory(trajectory_execution_request)
                        self.services.pause(std_srvs.srv.EmptyRequest())

                        local_env_params = self.planner.fwd_model.local_env_params
                        actual_path = ros_pycommon.trajectory_execution_response_to_numpy(traj_exec_response,
                                                                                          local_env_params,
                                                                                          self.services)
                        self.on_execution_complete(planned_path_dict,
                                                   planned_actions,
                                                   tail_goal,
                                                   actual_path,
                                                   full_env_data,
                                                   planner_data,
                                                   planning_time,
                                                   planner_status)

                    if self.pause_between_plans:
                        input("Press enter to proceed to next plan...")

                self.plan_idx += 1
                self.total_plan_idx += 1
                if self.plan_idx >= self.n_plans_per_env or self.total_plan_idx >= self.n_total_plans:
                    break

            if self.total_plan_idx >= self.n_total_plans:
                break

        self.on_complete(initial_poses_in_collision)

    def get_goal(self, w, h, head_point, env_padding, full_env_data):
        return sample_collision_free_goal(w, h, head_point, env_padding, full_env_data, self.goal_rng)

    def on_plan_complete(self,
                         planned_path: Dict[str, np.ndarray],
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_env_data: link_bot_sdf_utils.OccupancyData,
                         planner_data: ob.PlannerData,
                         planning_time: float,
                         planner_status: ob.PlannerStatus):
        pass

    def on_execution_complete(self,
                              planned_path: Dict[str, np.ndarray],
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              actual_path: Dict[str, np.ndarray],
                              full_env_data: link_bot_sdf_utils.OccupancyData,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: ob.PlannerStatus):
        pass

    def on_complete(self, initial_poses_in_collision):
        pass

    def on_planner_failure(self,
                           start: np.ndarray,
                           tail_goal_point: np.ndarray,
                           full_env_data: link_bot_sdf_utils.OccupancyData,
                           planner_data: ob.PlannerData):
        pass

    def on_after_plan(self):
        pass

    def on_before_plan(self):
        if self.sim_params.move_obstacles:
            # FIXME: instead of hard coding obstacles names, use the /objects service
            # generate a new environment by rearranging the obstacles
            objects = ['moving_box{}'.format(i) for i in range(1, 7)]
            gazebo_services.move_objects(self.services,
                                         self.sim_params.max_step_size,
                                         objects,
                                         self.planner.full_env_params.w,
                                         self.planner.full_env_params.h,
                                         padding=0.1,
                                         rng=self.env_rng)

