import numpy as np
import rospy
from colorama import Fore

from link_bot_gazebo.srv import ComputeSDF2Request, ComputeOccupancyRequest, ComputeSDF2, ComputeOccupancy, \
    LinkBotTrajectoryRequest

from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_pycommon import link_bot_sdf_utils


def get_n_state():
    return rospy.get_param("/link_bot/n_state")


def get_rope_length():
    return rospy.get_param("/link_bot/rope_length")


class Services:

    def __init__(self):
        self.compute_occupancy = rospy.ServiceProxy('/occupancy', ComputeOccupancy)

        # currently unused
        self.compute_sdf2 = rospy.ServiceProxy('/sdf2', ComputeSDF2)

        self.services_to_wait_for = [
            '/occupancy',
        ]

    def wait(self, verbose):
        if verbose >= 1:
            print(Fore.CYAN + "Waiting for services..." + Fore.RESET)
        for s in self.services_to_wait_for:
            rospy.wait_for_service(s)
        if verbose >= 1:
            print(Fore.CYAN + "Done waiting for services" + Fore.RESET)


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
    grid = np.array(response.grid).reshape([response.w_cols, response.h_rows])
    grid = grid.T
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
    return local_occupancy_data


def make_trajectory_execution_request(dt, actions):
    trajectory_execution_request = LinkBotTrajectoryRequest()
    trajectory_execution_request.dt = dt
    for action in actions:
        action_msg = LinkBotVelocityAction()
        action_msg.gripper1_velocity.x = action[0]
        action_msg.gripper1_velocity.y = action[1]
        trajectory_execution_request.gripper1_traj.append(action_msg)

    # NOTE: force stop at the end. Is this bad?
    stop_msg = LinkBotVelocityAction()
    stop_msg.gripper1_velocity.x = 0
    stop_msg.gripper1_velocity.y = 0
    trajectory_execution_request.gripper1_traj.append(stop_msg)

    return trajectory_execution_request


def trajectory_execution_response_to_numpy(trajectory_execution_result,
                                           local_env_params,
                                           services):
    actual_path = []
    actual_local_envs = []
    for configuration in trajectory_execution_result.actual_path:
        np_config = []
        for point in configuration.points:
            np_config.append(point.x)
            np_config.append(point.y)
        actual_path.append(np_config)

        actual_head_point = np.array([np_config[-2], np_config[-1]])
        if local_env_params is not None:
            actual_local_env = get_local_occupancy_data(rows=local_env_params.h_rows,
                                                        cols=local_env_params.w_cols,
                                                        res=local_env_params.res,
                                                        center_point=actual_head_point,
                                                        services=services)
            actual_local_envs.append(actual_local_env)
    actual_path = np.array(actual_path)
    return actual_path, actual_local_envs

