from time import sleep

import rospy
from colorama import Fore
from std_msgs.msg import String
from std_srvs.srv import Empty
from std_srvs.srv import EmptyRequest

from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
from gazebo_msgs.srv import GetPhysicsPropertiesRequest, SetPhysicsPropertiesRequest
from link_bot_gazebo.msg import MultiLinkBotPositionAction, LinkBotConfiguration, Position2dEnable, Position2dAction, \
    LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControl, LinkBotState


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
        self.get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        self.reset = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.services_to_wait_for = [
            '/world_control',
            '/link_bot_state',
            '/sdf',
            '/gazebo/get_physics_properties',
            '/gazebo/set_physics_properties',
            '/gazebo/reset_simulation',
            '/gazebo/pause_physics',
            '/gazebo/unpause_physics',
        ]

    def wait(self, args):
        if args.verbose:
            print(Fore.CYAN + "Waiting for services..." + Fore.RESET)
        for s in self.services_to_wait_for:
            rospy.wait_for_service(s)
        if args.verbose:
            print(Fore.CYAN + "Done waiting for services" + Fore.RESET)

    def reset_gazebo_environment(self):
        action_mode_msg = String()
        action_mode_msg.data = "velocity"

        stop_velocity_action = LinkBotVelocityAction()
        stop_velocity_action.gripper1_velocity.x = 0
        stop_velocity_action.gripper1_velocity.y = 0

        self.unpause(EmptyRequest())
        sleep(0.5)
        self.link_bot_mode.publish(action_mode_msg)
        self.velocity_action_pub.publish(stop_velocity_action)
        self.reset(EmptyRequest())
        sleep(0.5)
