from typing import Dict

import numpy as np

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
    occupancy_data = link_bot_sdf_utils.OccupancyData(data=environment['full_env/env'],
                                                      resolution=environment['full_env/res'],
                                                      origin=environment['full_env/origin'])
    occupancy_data = link_bot_sdf_utils.inflate(occupancy_data, radius_m=0.025)

    while True:
        x, y = sample_goal(goal_w_m, goal_h_m, rng)
        r, c = link_bot_sdf_utils.point_to_idx(x, y, resolution=occupancy_data.resolution, origin=occupancy_data.origin)
        collision = occupancy_data.data[r, c]
        if not collision:
            return x, y
