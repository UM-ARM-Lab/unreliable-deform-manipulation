from time import perf_counter
from typing import Dict, List

import numpy as np

from geometry_msgs.msg import Pose
from link_bot_pycommon import link_bot_sdf_utils
from peter_msgs.msg import LinkBotAction, ModelPose
from peter_msgs.srv import ComputeOccupancyRequest, LinkBotTrajectoryRequest


def get_occupancy(service_provider,
                  env_w_cols,
                  env_h_rows,
                  res,
                  center_x,
                  center_y,
                  robot_name):
    request = ComputeOccupancyRequest()
    request.resolution = res
    request.h_rows = env_h_rows
    request.w_cols = env_w_cols
    request.center.x = center_x
    request.center.y = center_y
    request.min_z = 0.05
    request.max_z = 2.00
    request.robot_name = robot_name
    request.request_new = True
    response = service_provider.compute_occupancy(request)
    grid = np.array(response.grid).reshape([response.w_cols, response.h_rows])
    grid = grid.T
    return grid, response


def get_occupancy_data(env_h_m,
                       env_w_m,
                       res,
                       service_provider,
                       robot_name):
    """
    :param env_h_m:  meters
    :param env_w_m: meters
    :param res: meters
    :param service_provider: from gazebo_utils
    :param robot_name: model name in gazebo
    :return:
    """
    env_h_rows = int(env_h_m / res)
    env_w_cols = int(env_w_m / res)
    grid, response = get_occupancy(service_provider,
                                   env_w_cols=env_w_cols,
                                   env_h_rows=env_h_rows,
                                   res=res,
                                   center_x=0,
                                   center_y=0,
                                   robot_name=robot_name)
    origin = np.array(response.origin)
    full_env_data = link_bot_sdf_utils.OccupancyData(data=grid, resolution=res, origin=origin)
    return full_env_data


def get_local_occupancy_data(rows,
                             cols,
                             res,
                             center_point,
                             service_provider,
                             robot_name):
    """
    :param rows: indices
    :param cols: indices
    :param res: meters
    :param center_point: (x,y) meters
    :param service_provider: from gazebo_utils
    :return: OccupancyData object for local sdf
    :param robot_name: model name in gazebo
    """
    grid, response = get_occupancy(service_provider,
                                   env_h_rows=rows,
                                   env_w_cols=cols,
                                   res=res,
                                   center_x=center_point[0],
                                   center_y=center_point[1],
                                   robot_name=robot_name)
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
            np_config = object.state_vector
            state[object.name] = np_config
        actual_path.append(state)

    return actual_path


def get_states_dict(service_provider, state_keys=None):
    start_states = {}
    objects_response = service_provider.get_objects()
    if state_keys is not None:
        for state_key in state_keys:
            for named_object in objects_response.objects.objects:
                if named_object.name == state_key:
                    state = np.array(named_object.state_vector)
                    start_states[state_key] = state
    else:
        # just take all of them
        for named_object in objects_response.objects.objects:
            start_states[named_object.name] = np.array(named_object.state_vector)

    return start_states


def xy_move(x: float, y: float):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 1
    return pose
