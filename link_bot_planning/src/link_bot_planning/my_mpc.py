#!/usr/bin/env python
from __future__ import division, print_function

import time
from typing import List

import numpy as np
import std_srvs
from colorama import Fore
from link_bot_gazebo.srv import LinkBotStateRequest
from ompl import base as ob

from ignition import markers
from link_bot_data import random_environment_data_utils
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning.goals import sample_goal
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import PlannerParams, SimParams
from link_bot_pycommon import link_bot_sdf_utils


class myMPC:

    def __init__(self,
                 planner: MyPlanner,
                 n_total_plans: int,
                 n_plans_per_env: int,
                 verbose: int,
                 planner_params: PlannerParams,
                 sim_params: SimParams,
                 services: GazeboServices,
                 no_execution: bool):
        self.planner = planner
        self.n_total_plans = n_total_plans
        self.n_plans_per_env = n_plans_per_env
        self.sim_params = sim_params
        self.planner_params = planner_params
        self.verbose = verbose
        self.services = services
        self.no_execution = no_execution

        # remove all markers
        markers.remove_all()

    def run(self):
        total_plan_idx = 0
        initial_poses_in_collision = 0
        while True:
            if self.sim_params.move_obstacles:
                # generate a new environment by rearranging the obstacles
                objects = ['moving_box{}'.format(i) for i in range(1, 7)]
                gazebo_utils.move_objects(self.services, self.sim_params.max_step_size, objects, self.planner.full_env_params.w,
                                          self.planner.full_env_params.h, 'velocity', padding=0.5)

            # nudge the rope so it is hopefully not in collision?
            self.services.nudge_rope(self.sim_params.max_step_size)

            # wait for things to settle
            # gazebo_utils.wait(duration_steps=100)

            # generate a bunch of plans to random goals
            state_req = LinkBotStateRequest()

            plan_idx = 0
            while True:
                # generate a random target
                state = self.services.get_state(state_req)
                head_idx = state.link_names.index("head")
                initial_rope_configuration = gazebo_utils.points_to_config(state.points)
                head_point = state.points[head_idx]
                tail_goal = self.get_goal(self.planner_params.w, self.planner_params.h, head_point,
                                          env_padding=self.sim_params.goal_padding)

                start = np.expand_dims(np.array(initial_rope_configuration), axis=0)
                tail_goal_point = np.array(tail_goal)

                # plan to that target
                if self.verbose >= 2:
                    # tail start x,y and tail goal x,y
                    random_environment_data_utils.publish_markers(tail_goal_point[0], tail_goal_point[1],
                                                                  initial_rope_configuration[0], initial_rope_configuration[1],
                                                                  marker_size=0.05)
                if self.verbose >= 1:
                    print(Fore.CYAN + "Planning from {} to {}".format(start, tail_goal_point) + Fore.RESET)

                t0 = time.time()
                try:
                    planner_results = self.planner.plan(start, tail_goal_point)
                    planned_actions, planned_path, planner_local_envs, full_env_data, planner_status = planner_results
                except RuntimeError:
                    # this means the start was considered invalid, so we just skip this and move to a new environment
                    print(Fore.RED + "Start was classified to be invalid. Skipping this environment." + Fore.RESET)
                    self.on_planner_failure(start, tail_goal_point, full_env_data)
                    initial_poses_in_collision += 1
                    break
                planning_time = time.time() - t0
                if self.verbose >= 1:
                    print("Planning time: {:5.3f}s".format(planning_time))

                planner_data = ob.PlannerData(self.planner.si)
                self.planner.planner.getPlannerData(planner_data)
                self.on_plan_complete(planned_path, tail_goal_point, planned_actions, full_env_data, planner_data, planning_time,
                                      planner_status)

                if self.verbose >= 4:
                    print("Planned actions: {}".format(planned_actions))
                    print("Planned path: {}".format(planned_path))

                trajectory_execution_request = gazebo_utils.make_trajectory_execution_request(self.planner.fwd_model.dt,
                                                                                              planned_actions)

                # execute the plan, collecting the states that actually occurred
                if not self.no_execution:
                    #  TODO: Consider executing just a few steps, so that our start states don't diverge too much
                    if self.verbose >= 2:
                        print(Fore.CYAN + "Executing Plan.".format(tail_goal_point) + Fore.RESET)

                    traj_exec_response = self.services.execute_trajectory(trajectory_execution_request)
                    self.services.pause(std_srvs.srv.EmptyRequest())

                    local_env_params = self.planner.fwd_model.local_env_params
                    actual_path, actual_local_envs = gazebo_utils.trajectory_execution_response_to_numpy(traj_exec_response,
                                                                                                         local_env_params,
                                                                                                         self.services)
                    self.on_execution_complete(planned_path,
                                               planned_actions,
                                               tail_goal_point,
                                               planner_local_envs,
                                               actual_local_envs,
                                               actual_path,
                                               full_env_data,
                                               planner_data,
                                               planning_time,
                                               planner_status)

                plan_idx += 1
                total_plan_idx += 1
                if plan_idx >= self.n_plans_per_env or total_plan_idx >= self.n_total_plans:
                    break

            if total_plan_idx >= self.n_total_plans:
                break

        self.on_complete(initial_poses_in_collision)

    def get_goal(self, w, h, head_point, env_padding):
        return sample_goal(w, h, head_point, env_padding)

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_env_data: link_bot_sdf_utils.OccupancyData,
                         planner_data: ob.PlannerData,
                         planning_time: float,
                         planner_status: ob.PlannerStatus):
        pass

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              planner_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_path: np.ndarray,
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
                           full_env_data: link_bot_sdf_utils.OccupancyData):
        pass
