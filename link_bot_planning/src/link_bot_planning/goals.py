import numpy as np

from link_bot_pycommon import link_bot_sdf_utils


def sample_goal(goal_w_m: float, goal_h_m: float, rng: np.random.RandomState):
    gripper1_target_x = rng.uniform(-goal_w_m / 2, goal_w_m / 2)
    gripper1_target_y = rng.uniform(-goal_h_m / 2, goal_h_m / 2)
    return gripper1_target_x, gripper1_target_y


def sample_collision_free_goal(goal_w_m: float,
                               goal_h_m: float,
                               full_env_data: link_bot_sdf_utils.OccupancyData,
                               rng: np.random.RandomState):
    """
    Args:
        goal_w_m: full width meters
        goal_h_m: full height meters
        full_env_data: occupancy data
        rng:  np rng
    Returns:
        x, y tuple, meters

    """
    full_env_data = link_bot_sdf_utils.inflate(full_env_data, radius_m=0.025)

    while True:
        x, y = sample_goal(goal_w_m, goal_h_m, rng)
        r, c = link_bot_sdf_utils.point_to_idx(x, y, full_env_data.resolution, full_env_data.origin)
        collision = full_env_data.data[r, c]
        if not collision:
            return x, y
