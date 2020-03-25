import numpy as np

from link_bot_pycommon import link_bot_sdf_utils


def sample_goal(w: float, h: float, rng: np.random.RandomState):
    gripper1_target_x = rng.uniform(-w / 2, w / 2)
    gripper1_target_y = rng.uniform(-h / 2, h / 2)
    return gripper1_target_x, gripper1_target_y


def sample_collision_free_goal(w: float,
                               h: float,
                               full_env_data: link_bot_sdf_utils.OccupancyData,
                               rng: np.random.RandomState):
    full_env_data = link_bot_sdf_utils.inflate(full_env_data, radius_m=0.05)

    while True:
        x, y = sample_goal(w, h, rng)
        r, c = link_bot_sdf_utils.point_to_idx(x, y, full_env_data.resolution, full_env_data.origin)
        collision = full_env_data.data[r, c]
        if not collision:
            return x, y
