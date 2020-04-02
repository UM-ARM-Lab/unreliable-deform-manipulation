from typing import Dict, List

import numpy as np
from peter_msgs.msg import LinkBotAction
from peter_msgs.srv import ComputeOccupancyRequest, LinkBotTrajectoryRequest

from link_bot_pycommon import link_bot_sdf_utils, link_bot_pycommon


def get_occupancy(service_provider,
                  env_w_cols,
                  env_h_rows,
                  res,
                  center_x,
                  center_y):
    request = ComputeOccupancyRequest()
    request.resolution = res
    request.h_rows = env_h_rows
    request.w_cols = env_w_cols
    request.center.x = center_x
    request.center.y = center_y
    request.min_z = 0.05
    request.max_z = 2.00
    request.robot_name = 'link_bot'
    request.request_new = True
    response = service_provider.compute_occupancy(request)
    grid = np.array(response.grid).reshape([response.w_cols, response.h_rows])
    grid = grid.T
    return grid, response


def get_occupancy_data(env_h_m,
                       env_w_m,
                       res,
                       service_provider):
    """
    :param env_h_m:  meters
    :param env_w_m: meters
    :param res: meters
    :param service_provider: from gazebo_utils
    :return:
    """
    env_h_rows = int(env_h_m / res)
    env_w_cols = int(env_w_m / res)
    grid, response = get_occupancy(service_provider,
                                   env_w_cols=env_w_cols,
                                   env_h_rows=env_h_rows,
                                   res=res,
                                   center_x=0,
                                   center_y=0)
    origin = np.array(response.origin)
    full_env_data = link_bot_sdf_utils.OccupancyData(data=grid, resolution=res, origin=origin)
    return full_env_data


def get_local_occupancy_data(rows,
                             cols,
                             res,
                             center_point,
                             service_provider):
    """
    :param rows: indices
    :param cols: indices
    :param res: meters
    :param center_point: (x,y) meters
    :param service_provider: from gazebo_utils
    :return: OccupancyData object for local sdf
    """
    grid, response = get_occupancy(service_provider,
                                   env_h_rows=rows,
                                   env_w_cols=cols,
                                   res=res,
                                   center_x=center_point[0],
                                   center_y=center_point[1])
    origin = np.array(response.origin)
    local_occupancy_data = link_bot_sdf_utils.OccupancyData(data=grid, resolution=res, origin=origin)
    return local_occupancy_data


def make_trajectory_execution_request(dt, actions):
    req = LinkBotTrajectoryRequest()
    for action in actions:
        action_msg = LinkBotAction()
        action_msg.max_time_per_step = dt
        action_msg.gripper1_delta_pos.x = action[0]
        action_msg.gripper1_delta_pos.y = action[1]
        req.gripper1_traj.append(action_msg)

    return req


def trajectory_execution_response_to_numpy(trajectory_execution_result) -> List[Dict[str, np.ndarray]]:
    actual_path = []
    for objects in trajectory_execution_result.actual_path:
        state = {}
        for object in objects.objects:
            np_config = link_bot_pycommon.flatten_named_points(object.points)
            state[object.name] = np_config
        actual_path.append(state)

    return actual_path


def get_start_states(service_provider, state_keys):
    start_states = {}
    objects_response = service_provider.get_objects()
    for state_key in state_keys:
        for object in objects_response.objects.objects:
            if object.name == state_key:
                state = link_bot_pycommon.flatten_named_points(object.points)
                start_states[state_key] = state

    return start_states
