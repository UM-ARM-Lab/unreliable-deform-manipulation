import numpy as np


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