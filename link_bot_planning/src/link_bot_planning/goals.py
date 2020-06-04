from typing import Dict

import numpy as np

import link_bot_pycommon.collision_checking
from link_bot_pycommon import link_bot_sdf_utils


def sample_goal(goal_w_m: float, goal_h_m: float, rng: np.random.RandomState):
    gripper1_target_x = rng.uniform(-goal_w_m / 2, goal_w_m / 2)
    gripper1_target_y = rng.uniform(-goal_h_m / 2, goal_h_m / 2)
    return gripper1_target_x, gripper1_target_y


def sample_collision_free_goal(goal_w_m: float,
                               goal_h_m: float,
                               environment: Dict,
                               rng: np.random.RandomState):
    """
    :param goal_w_m: full width meters
    :param goal_h_m: full height meters
    :param environment:
    :param rng:  np rng
    :return x, y tuple, meters
    """
    inflated_env = link_bot_pycommon.collision_checking.inflate_tf(environment['full_env/env'],
                                                                   radius_m=0.025,
                                                                   res=environment['full_env/res'])
    while True:
        x, y = sample_goal(goal_w_m, goal_h_m, rng)
        r, c = link_bot_sdf_utils.point_to_idx(x, y,
                                               resolution=environment['full_env/res'],
                                               origin=environment['full_env/origin'])
        collision = inflated_env[r, c]
        if not collision:
            return x, y
