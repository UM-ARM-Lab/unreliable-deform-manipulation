import numpy as np

from link_bot_pycommon import link_bot_sdf_utils


def sample_goal(w, h, current_head_point, env_padding):
    gripper1_current_x = current_head_point.x
    gripper1_current_y = current_head_point.y
    current = np.array([gripper1_current_x, gripper1_current_y])
    min_near = 0.5
    while True:
        gripper1_target_x = np.random.uniform(-w / 2 + env_padding, w / 2 - env_padding)
        gripper1_target_y = np.random.uniform(-h / 2 + env_padding, h / 2 - env_padding)
        target = np.array([gripper1_target_x, gripper1_target_y])
        d = np.linalg.norm(current - target)
        if d > min_near:
            break
    return gripper1_target_x, gripper1_target_y


def sample_collision_free_goal(w: float,
                               h: float,
                               current_head_point,
                               env_padding: float,
                               full_env_data: link_bot_sdf_utils.OccupancyData):
    full_env_data = link_bot_sdf_utils.inflate(full_env_data, radius_m=env_padding)

    while True:
        x, y = sample_goal(w, h, current_head_point, env_padding)
        r, c = link_bot_sdf_utils.point_to_idx(x, y, full_env_data.resolution, full_env_data.origin)
        collision = full_env_data.data[r, c]
        if not collision:
            return x, y
