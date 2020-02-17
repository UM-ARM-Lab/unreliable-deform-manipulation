#!/usr/bin/env python
import argparse

import rospy

from link_bot_gazebo.msg import LinkBotVelocityAction

from link_bot_gazebo.srv import ExecuteAction, ExecuteActionRequest
from link_bot_pycommon.args import my_formatter

if __name__ == '__main__':
    rospy.init_node('square')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('--size', type=float, default=0.15)
    parser.add_argument('--time', type=float, default=1.0)

    args = parser.parse_args()

    execute_action = rospy.ServiceProxy("/execute_action", ExecuteAction)

    action = ExecuteActionRequest()
    action.action.max_time_per_step = 1.0

    action.action.gripper1_delta_pos.x = args.size
    action.action.gripper1_delta_pos.y = 0
    execute_action(action)

    action.action.gripper1_delta_pos.x = 0
    action.action.gripper1_delta_pos.y = args.size
    execute_action(action)

    action.action.gripper1_delta_pos.x = -args.size
    action.action.gripper1_delta_pos.y = 0
    execute_action(action)

    action.action.gripper1_delta_pos.x = 0
    action.action.gripper1_delta_pos.y = -args.size
    execute_action(action)

    action.action.gripper1_delta_pos.x = 0
    action.action.gripper1_delta_pos.y = 0
    execute_action(action)
