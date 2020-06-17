from typing import Dict, List

import numpy as np

import rospy
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.base_services import Services
from link_bot_pycommon.link_bot_sdf_utils import extent_to_center, extent_to_env_shape
from peter_msgs.msg import LinkBotAction
from peter_msgs.srv import ComputeOccupancyRequest, LinkBotTrajectoryRequest, Position3DEnable, GetPosition3D, Position3DAction
from std_srvs.srv import Empty


def get_occupancy(service_provider,
                  env_w_cols,
                  env_h_rows,
                  env_c_channels,
                  res,
                  center_x,
                  center_y,
                  center_z,
                  robot_name):
    request = ComputeOccupancyRequest()
    request.resolution = res
    request.h_rows = env_h_rows
    request.w_cols = env_w_cols
    request.c_channels = env_c_channels
    request.center.x = center_x
    request.center.y = center_y
    request.center.z = center_z
    request.robot_name = robot_name
    request.request_new = True
    response = service_provider.compute_occupancy(request)
    grid = np.array(response.grid).reshape([env_w_cols, env_h_rows, env_c_channels])
    # this makes it so we can index with row, col, channel
    grid = np.transpose(grid, [1, 0, 2])
    return grid, response


def get_environment_for_extents_3d(extent,
                                   res: float,
                                   service_provider: Services,
                                   robot_name: str):
    cx, cy, cz = extent_to_center(extent)
    env_h_rows, env_w_cols, env_c_channels = extent_to_env_shape(extent, res)
    grid, response = get_occupancy(service_provider,
                                   env_w_cols=env_w_cols,
                                   env_h_rows=env_h_rows,
                                   env_c_channels=env_c_channels,
                                   res=res,
                                   center_x=cx,
                                   center_y=cy,
                                   center_z=cz,
                                   robot_name=robot_name)
    x_min = extent[0]
    y_min = extent[2]
    z_min = extent[4]
    origin_row = - x_min / res
    origin_col = -y_min / res
    origin_channel = -z_min / res
    origin = np.array([origin_row, origin_col, origin_channel], np.int32)
    return {
        'env': grid,
        'res': res,
        'origin': origin,
        'extent': extent,
    }


def get_occupancy_data(env_h_m: float,
                       env_w_m: float,
                       res: float,
                       service_provider: Services,
                       robot_name: str):
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
    env_c_channels = 1
    if robot_name is None:
        raise ValueError("robot name cannot be None")
    if robot_name == "":
        raise ValueError("robot name cannot be empty string")
    grid, response = get_occupancy(service_provider,
                                   env_w_cols=env_w_cols,
                                   env_h_rows=env_h_rows,
                                   env_c_channels=env_c_channels,
                                   res=res,
                                   center_x=0,
                                   center_y=0,
                                   center_z=res,  # we want to do a little off the ground because grid cells are centered
                                   robot_name=robot_name)
    origin = np.array(response.origin)
    full_env_data = link_bot_sdf_utils.OccupancyData(data=grid, resolution=res, origin=origin)
    return full_env_data


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


def make_movable_object_services(object_name):
    return {
        'enable': rospy.ServiceProxy(f'{object_name}/enable', Position3DEnable),
        'get_position': rospy.ServiceProxy(f'{object_name}/get', GetPosition3D),
        'action': rospy.ServiceProxy(f'{object_name}/set', Position3DAction),
        'stop': rospy.ServiceProxy(f'{object_name}/stop', Empty),
    }
