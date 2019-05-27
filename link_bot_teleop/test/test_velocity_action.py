import unittest

from time import sleep

import rospy
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest


class TestVelocityAction(unittest.TestCase):

    def test_velocity_action(self):
        """ send a configuration, then a velocity action, and check that it moved """
        rospy.init_node('test_velocity_action')

        config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
        config = LinkBotConfiguration()
        old_x = 2
        config.tail_pose.x = old_x
        config.tail_pose.y = 0
        config.tail_pose.theta = 0
        config.joint_angles_rad = [0, 0]
        config_pub.publish(config)

        sleep(1.0)

        action_pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=10)
        action_msg = LinkBotVelocityAction()
        action_msg.control_link_name = 'head'
        action_msg.vx = 1
        action_pub.publish(action_msg)

        sleep(1.0)

        world_control = rospy.ServiceProxy('/world_control', WorldControl)
        step = WorldControlRequest()
        step.steps = 1000  # assuming 0.001s of simulation time per step
        world_control.call(step)  # this will block until stepping is complete

        req = GetLinkStateRequest()
        req.link_name = "head"
        get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        response = get_link_state.call(req)
        new_x = response.link_state.pose.position.x

        self.assertNotAlmostEqual(new_x, old_x)
