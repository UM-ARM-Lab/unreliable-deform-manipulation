from time import sleep

import numpy as np
import rospy
from colorama import Fore
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyRequest
from std_srvs.srv import EmptyRequest

from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties, GetPhysicsPropertiesRequest, SetPhysicsPropertiesRequest
from link_bot_gazebo.msg import MultiLinkBotPositionAction, LinkBotConfiguration, Position2dEnable, Position2dAction, \
    LinkBotVelocityAction, ObjectAction
from link_bot_gazebo.srv import WorldControl, LinkBotState, ComputeSDF2, WorldControlRequest


class GazeboServices:

    def __init__(self):
        self.position_action_pub = rospy.Publisher("/multi_link_bot_position_action", MultiLinkBotPositionAction, queue_size=10)
        self.velocity_action_pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=10)
        self.config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10)
        self.link_bot_mode = rospy.Publisher('/link_bot_action_mode', String, queue_size=10)
        self.position_2d_enable = rospy.Publisher('/position_2d_enable', Position2dEnable, queue_size=10)
        self.position_2d_action = rospy.Publisher('/position_2d_action', Position2dAction, queue_size=10)
        self.world_control = rospy.ServiceProxy('/world_control', WorldControl)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.get_state = rospy.ServiceProxy('/link_bot_state', LinkBotState)
        self.compute_sdf = None
        self.compute_sdf2 = rospy.ServiceProxy('/sdf2', ComputeSDF2)
        self.get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        self.reset = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.services_to_wait_for = [
            '/world_control',
            '/link_bot_state',
            '/sdf',
            '/sdf2',
            '/gazebo/get_physics_properties',
            '/gazebo/set_physics_properties',
            '/gazebo/reset_simulation',
            '/gazebo/pause_physics',
            '/gazebo/unpause_physics',
        ]

    def wait(self, verbose=False):
        if verbose:
            print(Fore.CYAN + "Waiting for services..." + Fore.RESET)
        for s in self.services_to_wait_for:
            rospy.wait_for_service(s)
        if verbose:
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


def setup_gazebo_env(verbose, real_time_rate):
    # fire up services
    services = GazeboServices()
    services.wait(verbose)
    empty = EmptyRequest()
    services.reset.call(empty)
    # set up physics
    get = GetPhysicsPropertiesRequest()
    current_physics = services.get_physics.call(get)
    set = SetPhysicsPropertiesRequest()
    set.gravity = current_physics.gravity
    set.time_step = current_physics.time_step
    set.ode_config = current_physics.ode_config
    set.max_update_rate = real_time_rate * 1000.0
    set.enabled = True
    services.set_physics.call(set)
    # Set initial object positions
    move_action = Position2dAction()
    cheezits_move = ObjectAction()
    cheezits_move.pose.position.x = 0.20
    cheezits_move.pose.position.y = -0.25
    cheezits_move.pose.orientation.x = 0
    cheezits_move.pose.orientation.y = 0
    cheezits_move.pose.orientation.z = 0
    cheezits_move.pose.orientation.w = 0
    cheezits_move.model_name = "cheezits_box"
    move_action.actions.append(cheezits_move)
    tissue_move = ObjectAction()
    tissue_move.pose.position.x = 0.20
    tissue_move.pose.position.y = 0.25
    tissue_move.pose.orientation.x = 0
    tissue_move.pose.orientation.y = 0
    tissue_move.pose.orientation.z = 0
    tissue_move.pose.orientation.w = 0
    tissue_move.model_name = "tissue_box"
    move_action.actions.append(tissue_move)
    services.position_2d_action.publish(move_action)
    # let the simulator run to get the first image
    step = WorldControlRequest()
    step.steps = int(5.0 / 0.001)  # assuming 0.001s per simulation step
    services.world_control(step)  # this will block until stepping is complete
    return services
