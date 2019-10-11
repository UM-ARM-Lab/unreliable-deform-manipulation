#!/usr/bin/env python
from __future__ import division, print_function

import pathlib
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import std_srvs
from colorama import Fore
from dataclasses_json import dataclass_json

from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_data import random_environment_data_utils
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_gazebo.srv import LinkBotStateRequest, LinkBotTrajectoryRequest, LinkBotTrajectoryResponse
from link_bot_planning import classifier_utils, model_utils, shooting_rrt
from link_bot_planning.goals import sample_goal
from link_bot_pycommon import link_bot_sdf_utils
from visual_mpc import gazebo_trajectory_execution


# TODO: make external method for turning on and off visualization options (ros param server?, RQT GUI?)
@dataclass_json
@dataclass
class PlannerParams:
    timeout: float
    max_v: float


@dataclass_json
@dataclass
class EnvParams:
    w: float
    h: float
    real_time_rate: float
    goal_padding: float
    extent: List[float] = field(init=False)

    def __post_init__(self):
        self.extent = [-self.w, self.w, -self.h, self.h]


@dataclass_json
@dataclass
class SDFParams:
    full_h_m: float
    full_w_m: float
    local_h_rows: int
    local_w_cols: int
    res: float


class ShootingRRTMPC:

    def __init__(self,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 validator_model_dir: pathlib.Path,
                 validator_model_type: str,
                 n_envs: int,
                 n_targets_per_env: int,
                 verbose: int,
                 planner_params: PlannerParams,
                 sdf_params: SDFParams,
                 env_params: EnvParams,
                 services: GazeboServices,
                 ):
        self.fwd_model_dir = fwd_model_dir
        self.fwd_model_type = fwd_model_type
        self.validator_model_dir = validator_model_dir
        self.validator_model_type = validator_model_type
        self.n_envs = n_envs
        self.n_targets_per_env = n_targets_per_env
        self.sdf_params = sdf_params
        self.env_params = env_params
        self.planner_params = planner_params
        self.verbose = verbose
        self.services = services

        self.fwd_model = model_utils.load_generic_model(self.fwd_model_dir, self.fwd_model_type)
        self.classifier_model = NoneClassifier()
        # TODO: put this inside the generic model loader
        self.model_path_info = self.fwd_model_dir.parts[1:]
        self.validator_model = classifier_utils.load_generic_model(self.validator_model_dir, self.validator_model_type)

        self.rrt = shooting_rrt.ShootingRRT(fwd_model=self.fwd_model,
                                            classifier_model=self.classifier_model,
                                            dt=self.fwd_model.dt,
                                            n_state=self.fwd_model.n_state,
                                            planner_params=self.planner_params,
                                            sdf_params=sdf_params,
                                            env_params=env_params,
                                            services=services,
                                            )

    def run(self):
        for traj_idx in range(self.n_envs):
            # generate a new environment by rearranging the obstacles
            objects = ['moving_box{}'.format(i) for i in range(1, 7)]
            gazebo_trajectory_execution.move_objects(self.services, objects, self.env_params.w, self.env_params.h, 'velocity',
                                                     padding=0.5)

            # generate a bunch of plans to random goals
            state_req = LinkBotStateRequest()

            for plan_idx in range(self.n_targets_per_env):
                # generate a random target
                state = self.services.get_state(state_req)
                head_idx = state.link_names.index("head")
                rope_configuration = gazebo_utils.points_to_config(state.points)
                head_point = state.points[head_idx]
                tail_goal = sample_goal(self.env_params.w, self.env_params.h, head_point,
                                        env_padding=self.env_params.goal_padding)

                start = np.expand_dims(np.array(rope_configuration), axis=0)
                tail_goal_point = np.array(tail_goal)

                # plan to that target
                if self.verbose >= 2:
                    # tail start x,y and tail goal x,y
                    random_environment_data_utils.publish_markers(tail_goal_point[0], tail_goal_point[1],
                                                                  rope_configuration[0], rope_configuration[1],
                                                                  marker_size=0.05)
                if self.verbose >= 1:
                    print(Fore.CYAN + "Planning from {} to {}".format(start, tail_goal_point) + Fore.RESET)

                t0 = time.time()
                planned_actions, planned_path, _ = self.rrt.plan(start, tail_goal_point, self.verbose)
                planning_time = time.time() - t0
                self.on_plan_complete(planned_path, tail_goal_point, planned_actions, full_sdf_data, planning_time)

                trajectory_execution_request = LinkBotTrajectoryRequest()
                trajectory_execution_request.dt = self.fwd_model.dt
                if self.verbose >= 4:
                    print("Planned actions: {}".format(planned_actions))
                    print("Planned path: {}".format(planned_path))

                for action in planned_actions:
                    action_msg = LinkBotVelocityAction()
                    action_msg.gripper1_velocity.x = action[0]
                    action_msg.gripper1_velocity.y = action[1]
                    trajectory_execution_request.gripper1_traj.append(action_msg)

                # execute the plan, collecting the states that actually occurred
                #  TODO: Consider executing just a few steps, so that our start states don't diverge too much
                if self.verbose >= 2:
                    print(Fore.CYAN + "Executing Plan.".format(tail_goal_point) + Fore.RESET)

                trajectory_execution_result = self.services.execute_trajectory(trajectory_execution_request)
                self.services.pause(std_srvs.srv.EmptyRequest())

                self.on_execution_complete(planned_path,
                                           planned_actions,
                                           actual_local_sdfs,
                                           planner_local_sdfs,
                                           trajectory_execution_result)

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_sdf_data: link_bot_sdf_utils.SDF,
                         planning_time: float):
        pass

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              actual_local_sdfs: List[link_bot_sdf_utils.SDF],
                              planner_local_sdfs: List[link_bot_sdf_utils.SDF],
                              trajectory_execution_result: LinkBotTrajectoryResponse):
        pass
