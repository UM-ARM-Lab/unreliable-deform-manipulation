#!/usr/bin/env python
import argparse
import json
import pathlib

import colorama
import numpy as np

import ros_numpy
import rospy
from geometry_msgs.msg import Point
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("actions", type=pathlib.Path, help='actions json')

    args = parser.parse_args()

    rospy.init_node("replay_actions")

    with args.actions.open('r') as f:
        actions = json.load(f)

    action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)

    for action in actions:
        target_gripper1_point = ros_numpy.msgify(Point, np.array(action['gripper1_position']))
        target_gripper2_point = ros_numpy.msgify(Point, np.array(action['gripper2_position']))

        req = DualGripperTrajectoryRequest()
        req.gripper1_points.append(target_gripper1_point)
        req.gripper2_points.append(target_gripper2_point)
        action_srv(req)


if __name__ == "__main__":
    main()
