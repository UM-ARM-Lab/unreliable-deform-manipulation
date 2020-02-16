#!/usr/bin/env python
import argparse
from time import sleep

import rospy

from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_pycommon.args import my_formatter

if __name__ == '__main__':
    rospy.init_node('square')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('--speed', type=float, default=0.15)
    parser.add_argument('--time', type=float, default=1.0)

    args = parser.parse_args()

    pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=1)

    while pub.get_num_connections() < 1:
        pass

    action = LinkBotVelocityAction()

    action.gripper1_velocity.x = args.speed
    action.gripper1_velocity.y = 0
    pub.publish(action)
    sleep(args.time)

    action.gripper1_velocity.x = 0
    action.gripper1_velocity.y = args.speed
    pub.publish(action)
    sleep(args.time)

    action.gripper1_velocity.x = -args.speed
    action.gripper1_velocity.y = 0
    pub.publish(action)
    sleep(args.time)

    action.gripper1_velocity.x = -args.speed
    action.gripper1_velocity.y = 0
    pub.publish(action)
    sleep(args.time)

    action.gripper1_velocity.x = 0
    action.gripper1_velocity.y = 0
    pub.publish(action)
