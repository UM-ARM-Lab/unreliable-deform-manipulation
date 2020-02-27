from typing import Optional, Dict

import numpy as np
import rospy
import std_srvs
from colorama import Fore
from link_bot_gazebo.srv import ComputeSDF2Request, ComputeOccupancyRequest, ComputeSDF2, ComputeOccupancy, \
    LinkBotTrajectoryRequest, LinkBotState, WorldControl, LinkBotTrajectory, ExecuteAction, GetObjects

from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
from link_bot_gazebo.msg import LinkBotAction
from link_bot_pycommon import link_bot_sdf_utils, link_bot_pycommon

import matplotlib.pyplot as plt

def get_n_state():
    return rospy.get_param("/link_bot/n_state")


def get_n_tether_state():
    return rospy.get_param("/tether/n_state", default=0)


def get_rope_length():
    return rospy.get_param("/link_bot/rope_length")


def get_max_speed():
    return rospy.get_param("/link_bot/max_speed")


class EmptyRequest(object):
    pass


class Services:

    def __init__(self):
        self.compute_occupancy = rospy.ServiceProxy('/occupancy', ComputeOccupancy)
        self.get_state = rospy.ServiceProxy('/link_bot_state', LinkBotState)
        self.execute_action = rospy.ServiceProxy("/execute_action", ExecuteAction)
        self.world_control = rospy.ServiceProxy('/world_control', WorldControl)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', std_srvs.srv.Empty)
        self.execute_trajectory = rospy.ServiceProxy("/link_bot_execute_trajectory", LinkBotTrajectory)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', std_srvs.srv.Empty)
        self.get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        self.reset = rospy.ServiceProxy("/reset", std_srvs.srv.Empty)
        self.get_objects = rospy.ServiceProxy("/objects", GetObjects)

        # currently unused
        self.compute_sdf2 = rospy.ServiceProxy('/sdf2', ComputeSDF2)

        self.services_to_wait_for = [
            '/reset',
            '/world_control',
            '/link_bot_state',
            '/link_bot_execute_trajectory',
            '/occupancy',
            '/gazebo/pause_physics',
            '/gazebo/unpause_physics',
            '/gazebo/get_physics_properties',
            '/gazebo/set_physics_properties',
        ]

    def wait(self, verbose):
        if verbose >= 1:
            print(Fore.CYAN + "Waiting for services..." + Fore.RESET)
        for s in self.services_to_wait_for:
            rospy.wait_for_service(s)
        if verbose >= 1:
            print(Fore.CYAN + "Done waiting for services" + Fore.RESET)

    @staticmethod
    def setup_env(verbose: int,
                  real_time_rate: float,
                  reset_gripper_to: Optional,
                  max_step_size: Optional[float] = None,
                  initial_object_dict: Optional[Dict] = None):
        pass


def get_sdf_and_gradient(services,
                         env_w_cols,
                         env_h_rows,
                         res,
                         center_x,
                         center_y):
    sdf_request = ComputeSDF2Request()
    sdf_request.resolution = res
    sdf_request.h_rows = env_h_rows
    sdf_request.w_cols = env_w_cols
    sdf_request.center.x = center_x
    sdf_request.center.y = center_y
    sdf_request.min_z = 0.01
    sdf_request.max_z = 2.00
    sdf_request.robot_name = 'link_bot'
    sdf_request.request_new = True
    sdf_response = services.compute_sdf2(sdf_request)
    sdf = np.array(sdf_response.sdf).reshape([sdf_response.gradient.layout.dim[0].size, sdf_response.gradient.layout.dim[1].size])
    sdf = sdf.T
    gradient = np.array(sdf_response.gradient.data).reshape([sdf_response.gradient.layout.dim[0].size,
                                                             sdf_response.gradient.layout.dim[1].size,
                                                             sdf_response.gradient.layout.dim[2].size])
    gradient = np.transpose(gradient, [1, 0, 2])

    return gradient, sdf, sdf_response


def get_sdf_data(env_h,
                 env_w,
                 res,
                 services):
    """
    :param env_h:  meters
    :param env_w: meters
    :param res: meters
    :param services: from gazebo_utils
    :return: SDF object for full sdf
    """
    env_h_rows = int(env_h / res)
    env_w_cols = int(env_w / res)
    gradient, sdf, sdf_response = get_sdf_and_gradient(services, env_w_cols=env_w_cols, env_h_rows=env_h_rows, res=res)
    resolution = np.array(sdf_response.res)
    origin = np.array(sdf_response.origin)
    full_sdf_data = link_bot_sdf_utils.SDF(sdf=sdf, gradient=gradient, resolution=resolution, origin=origin)
    return full_sdf_data


def get_occupancy(services,
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
    request.min_z = 0.1  # FIXME: maybe lower this again?
    request.max_z = 2.00
    request.robot_name = 'link_bot'
    request.request_new = True
    response = services.compute_occupancy(request)
    grid = np.array(response.grid).reshape([response.h_rows, response.w_cols])
    grid = grid.T
    # import ipdb; ipdb.set_trace()
    return grid, response


def get_occupancy_data(env_h,
                       env_w,
                       res,
                       services):
    """
    :param env_h:  meters
    :param env_w: meters
    :param res: meters
    :param services: from gazebo_utils
    :return: SDF object for full sdf
    """
    env_h_rows = int(env_h / res)
    env_w_cols = int(env_w / res)
    grid, response = get_occupancy(services,
                                   env_w_cols=env_w_cols,
                                   env_h_rows=env_h_rows,
                                   res=res,
                                   center_x=0,
                                   center_y=0)
    resolution = np.array(response.res)
    origin = np.array(response.origin)
    full_env_data = link_bot_sdf_utils.OccupancyData(data=grid, resolution=resolution, origin=origin)
    return full_env_data


def get_local_sdf_data(sdf_rows,
                       sdf_cols,
                       res,
                       center_point,
                       services):
    """
    :param sdf_rows: indices
    :param sdf_cols: indices
    :param res: meters
    :param center_point: (x,y) meters
    :param services: from gazebo_utils
    :return: SDF object for local sdf
    """
    gradient, sdf, sdf_response = get_sdf_and_gradient(services,
                                                       env_h_rows=sdf_rows,
                                                       env_w_cols=sdf_cols,
                                                       res=res,
                                                       center_x=center_point[0],
                                                       center_y=center_point[1])
    resolution = np.array(sdf_response.res)
    origin = np.array(sdf_response.origin)
    local_sdf_data = link_bot_sdf_utils.SDF(sdf=sdf, gradient=gradient, resolution=resolution, origin=origin)
    return local_sdf_data


def get_local_occupancy_data(rows,
                             cols,
                             res,
                             center_point,
                             services):
    """
    :param rows: indices
    :param cols: indices
    :param res: meters
    :param center_point: (x,y) meters
    :param services: from gazebo_utils
    :return: OccupancyData object for local sdf
    """
    grid, response = get_occupancy(services,
                                   env_h_rows=rows,
                                   env_w_cols=cols,
                                   res=res,
                                   center_x=center_point[0],
                                   center_y=center_point[1])
    resolution = np.array(response.res)
    origin = np.array(response.origin)
    local_occupancy_data = link_bot_sdf_utils.OccupancyData(data=grid, resolution=resolution, origin=origin)
    # import ipdb; ipdb.set_trace()
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


def trajectory_execution_response_to_numpy(trajectory_execution_result,
                                           local_env_params,
                                           services):
    actual_path = {
        'local_env': [],
        'local_env_origin': [],
    }

    for objects in trajectory_execution_result.actual_path:
        for object in objects.objects:
            if object.name not in actual_path:
                actual_path[object.name] = []

            np_config = []
            for named_point in object.points:
                np_config.append(named_point.point.x)
                np_config.append(named_point.point.y)

            actual_path[object.name].append(np_config)

            if object.name == 'link_bot' and local_env_params is not None:
                actual_head_point = np.array([np_config[-2], np_config[-1]])
                actual_local_env = get_local_occupancy_data(rows=local_env_params.h_rows,
                                                            cols=local_env_params.w_cols,
                                                            res=local_env_params.res,
                                                            center_point=actual_head_point,
                                                            services=services)
                actual_path['local_env'].append(actual_local_env.data)
                actual_path['local_env_origin'].append(actual_local_env.origin)

    for k, v in actual_path.items():
        actual_path[k] = np.array(v)

    return actual_path


def get_start_states(services, state_keys):
    start_states = {}
    objects_response = services.get_objects()
    link_bot_start_state = None
    head_point = None
    for subspace_name in state_keys:
        for object in objects_response.objects.objects:
            if object.name == subspace_name:
                state = link_bot_pycommon.flatten_named_points(object.points)
                if subspace_name == 'link_bot':
                    link_bot_start_state = state
                    head_point = link_bot_pycommon.get_head_from_named_points(object.points)
                start_states[subspace_name] = state
    start_states['link_bot'] = link_bot_start_state
    return start_states, link_bot_start_state, head_point
