from time import sleep
from typing import Optional, Dict, Iterable, List

import numpy as np
import rospy
import std_msgs
import std_srvs
from colorama import Fore
from link_bot_gazebo.msg import *
from link_bot_gazebo.msg import *
from link_bot_gazebo.srv import *
from std_msgs.msg import String, Empty
from std_srvs.srv import EmptyRequest

from gazebo_msgs.srv import *
from link_bot_planning.params import LocalEnvParams
from link_bot_pycommon import link_bot_sdf_utils
from visual_mpc import sensor_image_to_float_image
from visual_mpc.gazebo_trajectory_execution import quaternion_from_euler
from visual_mpc.numpy_point import NumpyPoint


def points_to_config(points):
    return np.array([[p.x, p.y] for p in points]).flatten()


class GazeboServices:

    def __init__(self):
        self.velocity_action_pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=10)
        self.config_pub = rospy.Publisher('/link_bot_configuration', LinkBotJointConfiguration, queue_size=10)
        self.link_bot_mode = rospy.Publisher('/link_bot_action_mode', String, queue_size=10)
        self.position_2d_stop = rospy.Publisher('/position_2d_stop', std_msgs.msg.Empty, queue_size=10)
        self.position_2d_action = rospy.Publisher('/position_2d_action', Position2dAction, queue_size=10)
        self.world_control = rospy.ServiceProxy('/world_control', WorldControl)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', std_srvs.srv.Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', std_srvs.srv.Empty)
        self.get_state = rospy.ServiceProxy('/link_bot_state', LinkBotState)
        self.compute_occupancy = rospy.ServiceProxy('/occupancy', ComputeOccupancy)
        self.get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        self.reset = rospy.ServiceProxy("/gazebo/reset_simulation", std_srvs.srv.Empty)
        self.execute_trajectory = rospy.ServiceProxy("/link_bot_execute_trajectory", LinkBotTrajectory)
        self.apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        # currently unused
        self.compute_sdf2 = rospy.ServiceProxy('/sdf2', ComputeSDF2)
        self.position_action = rospy.ServiceProxy("/link_bot_position_action", LinkBotPositionAction)
        self.execute_path = rospy.ServiceProxy("/link_bot_execute_path", LinkBotPath)
        self.xy_to_rowcol = rospy.ServiceProxy('/my_camera/xy_to_rowcol', CameraProjection)
        self.rowcol_to_xy = rospy.ServiceProxy('/my_camera/rowcol_to_xy', InverseCameraProjection)

        self.services_to_wait_for = [
            '/world_control',
            '/link_bot_state',
            '/link_bot_position_action',
            '/link_bot_execute_path',
            '/link_bot_execute_trajectory',
            '/sdf',
            '/sdf2',
            '/gazebo/get_physics_properties',
            '/gazebo/set_physics_properties',
            '/gazebo/reset_simulation',
            '/gazebo/pause_physics',
            '/gazebo/unpause_physics',
            '/my_camera/xy_to_rowcol',
            '/my_camera/rowcol_to_xy',
        ]

    def wait(self, verbose: int = 0):
        if verbose >= 1:
            print(Fore.CYAN + "Waiting for services..." + Fore.RESET)
        for s in self.services_to_wait_for:
            rospy.wait_for_service(s)
        if verbose >= 1:
            print(Fore.CYAN + "Done waiting for services" + Fore.RESET)

    def reset_gazebo_environment(self, reset_model_poses=True):
        action_mode_msg = String()
        action_mode_msg.data = "velocity"

        stop_velocity_action = LinkBotVelocityAction()
        stop_velocity_action.gripper1_velocity.x = 0
        stop_velocity_action.gripper1_velocity.y = 0

        self.unpause(EmptyRequest())
        sleep(0.5)
        self.link_bot_mode.publish(action_mode_msg)
        self.velocity_action_pub.publish(stop_velocity_action)
        if reset_model_poses:
            self.reset(EmptyRequest())
        sleep(0.5)

    def get_context(self, context_length, state_dim, action_dim, image_h=64, image_w=64, image_d=3):
        # TODO: don't require these dimensions as arguments, they can be figured out from the messages/services
        state_req = LinkBotStateRequest()
        initial_context_images = np.ndarray((context_length, image_h, image_w, image_d))
        initial_context_states = np.ndarray((context_length, state_dim))
        for t in range(context_length):
            state = self.get_state.call(state_req)
            # FIXME: wtf is this shit
            if state_dim == 6:
                rope_config = points_to_config(state.points)
            else:
                rope_config = np.array([state.points[-1].x, state.points[-1].y])
            # Convert to float image
            image = sensor_image_to_float_image(state.camera_image.data, image_h, image_w, image_d)
            initial_context_images[t] = image
            initial_context_states[t] = rope_config

        context_images = initial_context_images
        context_states = initial_context_states
        context_actions = np.zeros([context_length - 1, action_dim])
        return context_images, context_states, context_actions

    def nudge_rope(self, max_step_size: float):
        nudge = ApplyBodyWrenchRequest()
        nudge.duration.secs = 0
        nudge.duration.nsecs = 50000000
        nudge.body_name = 'link_bot::head'
        angle = np.random.uniform(-np.pi, np.pi)
        magnitude = 10  # newtons
        nudge.wrench.force.x = np.cos(angle) * magnitude
        nudge.wrench.force.y = np.sin(angle) * magnitude

        self.apply_body_wrench(nudge)

        wait = WorldControlRequest()
        wait.steps = int(5 / max_step_size)  # assuming 0.001s per simulation step
        self.world_control(wait)


def rowcol_to_xy(services, row, col):
    req = InverseCameraProjectionRequest()
    req.rowcol.x_col = col
    req.rowcol.y_row = row
    res = services.rowcol_to_xy(req)
    return res.xyz.x, res.xyz.y


def setup_gazebo_env(verbose: int,
                     real_time_rate: float,
                     max_step_size: Optional[float] = None,
                     reset_world: Optional[bool] = True,
                     initial_object_dict: Optional[Dict] = None) -> GazeboServices:
    # fire up services
    services = GazeboServices()
    services.wait(verbose)

    if reset_world:
        empty = EmptyRequest()
        services.reset.call(empty)

    # set up physics
    get = GetPhysicsPropertiesRequest()
    current_physics = services.get_physics.call(get)
    set = SetPhysicsPropertiesRequest()
    set.gravity = current_physics.gravity
    set.ode_config = current_physics.ode_config
    set.max_update_rate = real_time_rate * 1000.0
    if max_step_size is None:
        max_step_size = current_physics.time_step
    set.time_step = max_step_size
    set.enabled = True
    services.set_physics.call(set)

    # Set initial object positions
    if initial_object_dict is not None:
        move_action = Position2dAction()
        for object_name, (x, y) in initial_object_dict.items():
            move = ObjectAction()
            move.pose.position.x = x
            move.pose.position.y = y
            move.pose.orientation.x = 0
            move.pose.orientation.y = 0
            move.pose.orientation.z = 0
            move.pose.orientation.w = 0
            move.model_name = object_name
            move_action.actions.append(move)
        services.position_2d_action.publish(move_action)

    services.position_2d_stop.publish(Empty())
    return services


def get_rope_head_pixel_coordinates(services):
    state_req = LinkBotStateRequest()
    state = services.get_state(state_req)
    head_x_m = state.points[-1].x
    head_y_m = state.points[-1].y
    req = CameraProjectionRequest()
    req.xyz.x = head_x_m
    req.xyz.y = head_y_m
    req.xyz.z = 0.01
    res = services.xy_to_rowcol(req)
    return NumpyPoint(int(res.rowcol.x_col), int(res.rowcol.y_row)), head_x_m, head_y_m


def xy_to_row_col(services, x, y, z):
    req = CameraProjectionRequest()
    req.xyz.x = x
    req.xyz.y = y
    req.xyz.z = z
    res = services.xy_to_rowcol(req)
    return NumpyPoint(int(res.rowcol.x_col), int(res.rowcol.y_row))


def get_occupancy(services: GazeboServices,
                  env_w_cols: int,
                  env_h_rows: int,
                  res: float = 0.01,
                  center_x: float = 0,
                  center_y: float = 0):
    request = ComputeOccupancyRequest()
    request.resolution = res
    request.h_rows = env_h_rows
    request.w_cols = env_w_cols
    request.center.x = center_x
    request.center.y = center_y
    request.min_z = 0.01
    request.max_z = 2.00
    request.robot_name = 'link_bot'
    request.request_new = True
    response = services.compute_occupancy(request)
    grid = np.array(response.grid).reshape([response.w_cols, response.h_rows])
    grid = grid.T
    return grid, response


def get_sdf_and_gradient(services: GazeboServices,
                         env_w_cols: int,
                         env_h_rows: int,
                         res: float = 0.01,
                         center_x: float = 0,
                         center_y: float = 0):
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


def get_sdf_data(env_h: float,
                 env_w: float,
                 res: float,
                 services: GazeboServices):
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


def get_occupancy_data(env_h: float,
                       env_w: float,
                       res: float,
                       services: GazeboServices):
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


def get_local_sdf_data(sdf_rows: int,
                       sdf_cols: int,
                       res: float,
                       center_point: np.ndarray,
                       services: GazeboServices):
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


def get_local_occupancy_data(rows: int,
                             cols: int,
                             res: float,
                             center_point: np.ndarray,
                             services: GazeboServices) -> link_bot_sdf_utils.OccupancyData:
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


def make_trajectory_execution_request(dt: float, actions: Iterable) -> LinkBotTrajectoryRequest:
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


def trajectory_execution_response_to_numpy(trajectory_execution_result: LinkBotTrajectoryResponse,
                                           local_env_params: Optional[LocalEnvParams],
                                           services: GazeboServices) -> [np.ndarray, List[link_bot_sdf_utils.OccupancyData]]:
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


def random_object_move(model_name, w, h, padding):
    move = ObjectAction()
    move.pose.position.x = np.random.uniform(-w / 2 + padding, w / 2 - padding)
    move.pose.position.y = np.random.uniform(-h / 2 + padding, h / 2 - padding)
    q = quaternion_from_euler(0, 0, np.random.uniform(-np.pi, np.pi))
    move.pose.orientation.x = q[0]
    move.pose.orientation.y = q[1]
    move.pose.orientation.z = q[2]
    move.pose.orientation.w = q[3]
    move.model_name = model_name
    return move


def move_objects(services, max_step_size, objects, env_w, env_h, link_bot_mode, padding):
    disable_link_bot = String()
    disable_link_bot.data = 'disabled'

    enable_link_bot = String()
    enable_link_bot.data = link_bot_mode

    # disable the rope controller, enable the objects
    services.link_bot_mode.publish(disable_link_bot)
    # Move the objects
    move_action = Position2dAction()
    for object_name in objects:
        move = random_object_move(object_name, env_w, env_h, padding)
        move_action.actions.append(move)
    services.position_2d_action.publish(move_action)
    # let the move actually occur
    step = WorldControlRequest()
    move_wait_duration = 1
    step.steps = int(move_wait_duration / max_step_size)
    services.world_control(step)  # this will block until stepping is complete
    # disable the objects so they stop, enabled the rope controller
    services.position_2d_stop.publish(Empty())
    services.link_bot_mode.publish(enable_link_bot)


def get_n_state():
    return rospy.get_param("/link_bot/n_state")


def get_rope_length():
    return rospy.get_param("/link_bot/rope_length")
