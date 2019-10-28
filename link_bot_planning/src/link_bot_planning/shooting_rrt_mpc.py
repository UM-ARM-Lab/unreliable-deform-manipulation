#!/usr/bin/env python
from __future__ import division, print_function

import pathlib
import time
from typing import List

import numpy as np
import std_srvs
from colorama import Fore
from ompl import base as ob

from link_bot_data import random_environment_data_utils
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices, get_sdf_data
from link_bot_gazebo.srv import LinkBotStateRequest
from link_bot_planning import classifier_utils, model_utils, shooting_rrt, ompl_viz
from link_bot_planning.goals import sample_goal
from link_bot_planning.params import PlannerParams, LocalEnvParams, EnvParams
from link_bot_pycommon import link_bot_sdf_utils
from visual_mpc import gazebo_trajectory_execution


class ShootingRRTMPC:

    def __init__(self,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 classifier_model_dir: pathlib.Path,
                 classifier_model_type: str,
                 n_envs: int,
                 n_targets_per_env: int,
                 verbose: int,
                 planner_params: PlannerParams,
                 local_env_params: LocalEnvParams,
                 env_params: EnvParams,
                 services: GazeboServices,
                 ):
        self.fwd_model_dir = fwd_model_dir
        self.fwd_model_type = fwd_model_type
        self.classifier_model_dir = classifier_model_dir
        self.classifier_model_type = classifier_model_type
        self.n_envs = n_envs
        self.n_targets_per_env = n_targets_per_env
        self.local_env_params = local_env_params
        self.env_params = env_params
        self.planner_params = planner_params
        self.verbose = verbose
        self.services = services

        self.fwd_model, self.model_path_info = model_utils.load_generic_model(self.fwd_model_dir, self.fwd_model_type)
        self.classifier_model = classifier_utils.load_generic_model(self.classifier_model_dir, self.classifier_model_type)
        self.viz_object = ompl_viz.VizObject()

        self.rrt = shooting_rrt.ShootingRRT(fwd_model=self.fwd_model,
                                            classifier_model=self.classifier_model,
                                            dt=self.fwd_model.dt,
                                            n_state=self.fwd_model.n_state,
                                            planner_params=self.planner_params,
                                            local_env_params=local_env_params,
                                            env_params=env_params,
                                            services=services,
                                            viz_object=self.viz_object,
                                            )

    def run(self):
        for traj_idx in range(self.n_envs):
            # generate a new environment by rearranging the obstacles
            objects = ['moving_box{}'.format(i) for i in range(1, 7)]
            gazebo_trajectory_execution.move_objects(self.services, objects, self.env_params.w, self.env_params.h, 'velocity',
                                                     padding=0.5)

            # TODO: should I have this here? It's just for visualization
            full_sdf_data = get_sdf_data(env_h=10, env_w=10, res=0.03, services=self.services)

            # generate a bunch of plans to random goals
            state_req = LinkBotStateRequest()

            for plan_idx in range(self.n_targets_per_env):
                # generate a random target
                state = self.services.get_state(state_req)
                head_idx = state.link_names.index("head")
                initial_rope_configuration = gazebo_utils.points_to_config(state.points)
                head_point = state.points[head_idx]
                tail_goal = sample_goal(self.env_params.w, self.env_params.h, head_point,
                                        env_padding=self.env_params.goal_padding)

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
                planned_actions, planned_path, planner_local_envs = self.rrt.plan(start, tail_goal_point)
                planning_time = time.time() - t0
                if self.verbose >= 1:
                    print("Planning time: {:5.3f}s".format(planning_time))

                planner_data = ob.PlannerData(self.rrt.si)
                self.rrt.planner.getPlannerData(planner_data)
                self.on_plan_complete(planned_path, tail_goal_point, planned_actions, full_sdf_data, planner_data, planning_time)

                if self.verbose >= 4:
                    print("Planned actions: {}".format(planned_actions))
                    print("Planned path: {}".format(planned_path))

                trajectory_execution_request = gazebo_utils.make_trajectory_execution_request(self.fwd_model.dt, planned_actions)

                # execute the plan, collecting the states that actually occurred
                #  TODO: Consider executing just a few steps, so that our start states don't diverge too much
                if self.verbose >= 2:
                    print(Fore.CYAN + "Executing Plan.".format(tail_goal_point) + Fore.RESET)

                traj_exec_response = self.services.execute_trajectory(trajectory_execution_request)
                self.services.pause(std_srvs.srv.EmptyRequest())

                actual_path, actual_local_envs = gazebo_utils.trajectory_execution_response_to_numpy(traj_exec_response,
                                                                                                     self.local_env_params,
                                                                                                     self.services)
                self.on_execution_complete(planned_path,
                                           planned_actions,
                                           planner_local_envs,
                                           actual_local_envs,
                                           actual_path)

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_sdf_data: link_bot_sdf_utils.SDF,
                         planner_data: ob.PlannerData,
                         planning_time: float):
        pass

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              planner_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_path: np.ndarray):
        pass
