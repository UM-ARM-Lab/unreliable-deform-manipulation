#!/usr/bin/env python

import numpy as np
import rospy
from link_bot_gazebo.msg import LinkBotConfiguration
from sensor_msgs.msg import Joy

if __name__ == '__main__':
    rospy.init_node("reset_link_bot")
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    joy_pub = rospy.Publisher('/joy', Joy, queue_size=10, latch=True)
    config = LinkBotConfiguration()
    config.tail_pose.x = 0
    config.tail_pose.y = 0
    config.tail_pose.theta = np.pi / 2
    config.joint_angles_rad = [0, 0]
    config_pub.publish(config)
    j = Joy()
    j.axes = [0, 0]
    joy_pub.publish(j)
    rospy.spin()
