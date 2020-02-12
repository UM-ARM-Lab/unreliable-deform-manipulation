from time import sleep
from typing import Optional, Dict

import numpy as np
import rospy
import std_msgs
import std_srvs
from link_bot_gazebo.msg import LinkBotVelocityAction, LinkBotJointConfiguration, Position2dAction, ObjectAction
from link_bot_gazebo.srv import LinkBotPositionAction, LinkBotPath, \
    CameraProjection, InverseCameraProjection, LinkBotStateRequest, WorldControlRequest, InverseCameraProjectionRequest, \
    CameraProjectionRequest
from std_msgs.msg import String, Empty
from std_srvs.srv import EmptyRequest

from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest, \
    SetPhysicsPropertiesRequest, GetPhysicsPropertiesRequest
from link_bot_pycommon.link_bot_pycommon import points_to_config
from link_bot_pycommon.ros_pycommon import Services
from visual_mpc import sensor_image_to_float_image
from visual_mpc.gazebo_trajectory_execution import quaternion_from_euler
from visual_mpc.numpy_point import NumpyPoint


class GazeboServices(Services):

    def __init__(self):
        super().__init__()
        self.reset = rospy.ServiceProxy("/gazebo/reset_simulation", std_srvs.srv.Empty)

        # we can't mock these
        self.apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        # FIXME: wrap up all this "automatically rearrange the environment" business
        self.link_bot_mode = rospy.Publisher('/link_bot_action_mode', String, queue_size=10)
        self.position_2d_stop = rospy.Publisher('/position_2d_stop', std_msgs.msg.Empty, queue_size=10)
        self.position_2d_action = rospy.Publisher('/position_2d_action', Position2dAction, queue_size=10)

        # currently unused
        self.position_action = rospy.ServiceProxy("/link_bot_position_action", LinkBotPositionAction)
        self.execute_path = rospy.ServiceProxy("/link_bot_execute_path", LinkBotPath)
        self.xy_to_rowcol = rospy.ServiceProxy('/my_camera/xy_to_rowcol', CameraProjection)
        self.rowcol_to_xy = rospy.ServiceProxy('/my_camera/rowcol_to_xy', InverseCameraProjection)
        self.config_pub = rospy.Publisher('/link_bot_configuration', LinkBotJointConfiguration, queue_size=10)

        self.services_to_wait_for.extend([
            '/gazebo/reset_simulation',
        ])

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

    def nudge_rope(self, max_step_size: float, rng: np.random.RandomState):
        nudge = ApplyBodyWrenchRequest()
        nudge.duration.secs = 0
        nudge.duration.nsecs = 50000000
        nudge.body_name = 'link_bot::head'
        angle = rng.uniform(-np.pi, np.pi)
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


def random_object_move(model_name: str, w: float, h: float, padding: float, rng: np.random.RandomState):
    move = ObjectAction()
    move.pose.position.x = rng.uniform(-w / 2 + padding, w / 2 - padding)
    move.pose.position.y = rng.uniform(-h / 2 + padding, h / 2 - padding)
    q = quaternion_from_euler(0, 0, rng.uniform(-np.pi, np.pi))
    move.pose.orientation.x = q[0]
    move.pose.orientation.y = q[1]
    move.pose.orientation.z = q[2]
    move.pose.orientation.w = q[3]
    move.model_name = model_name
    return move


def move_objects(services,
                 max_step_size: float,
                 objects,
                 env_w: float,
                 env_h: float,
                 link_bot_mode,
                 padding: float,
                 rng: np.random.RandomState):
    disable_link_bot = String()
    disable_link_bot.data = 'disabled'

    enable_link_bot = String()
    enable_link_bot.data = link_bot_mode

    # disable the rope controller, enable the objects
    services.link_bot_mode.publish(disable_link_bot)
    # Move the objects
    move_action = Position2dAction()
    for object_name in objects:
        move = random_object_move(object_name, env_w, env_h, padding, rng)
        move_action.actions.append(move)
    services.position_2d_action.publish(move_action)
    # let the move actually occur
    step = WorldControlRequest()
    move_wait_duration = 0.75
    step.steps = int(move_wait_duration / max_step_size)
    services.world_control(step)  # this will block until stepping is complete
    # disable the objects so they stop, enabled the rope controller
    services.position_2d_stop.publish(Empty())
    services.link_bot_mode.publish(enable_link_bot)

    # wait a few steps to ensure the stop message is received
    wait = WorldControlRequest()
    wait.steps = int(2 / max_step_size)
    services.world_control(wait)  # this will block until stepping is complete
