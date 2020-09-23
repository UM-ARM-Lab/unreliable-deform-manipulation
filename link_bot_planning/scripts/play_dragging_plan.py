#!/usr/bin/env python
import argparse
import gzip
import json
import pathlib

import colorama
import numpy as np

import ros_numpy
import rospy
from geometry_msgs.msg import Point
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("--trial-idx", type=int, help='which plan to show', default=0)

    rospy.init_node("play_dragging_plan")

    action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)

    args = parser.parse_args()

    with (args.results_dir / 'metadata.json').open('r') as metadata_file:
        metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)
    scenario = get_scenario(metadata['scenario'])

    with gzip.open(args.results_dir / f'{args.trial_idx}_metrics.json.gz', 'rb') as metrics_file:
        metrics_str = metrics_file.read()
    datum = json.loads(metrics_str.decode("utf-8"))

    all_actions = []
    steps = datum['steps']
    for step_idx, step in enumerate(steps):
        if step['type'] == 'executed_plan':
            actions = step['planning_result']['actions']
        elif step['type'] == 'executed_recovery':
            actions = [step['recovery_action']]

        all_actions.extend(actions)

    for action in all_actions:
        target_gripper1_point = ros_numpy.msgify(Point, np.array(action['gripper_position']))
        target_gripper1_point.z = -0.39
        target_gripper2_point = ros_numpy.msgify(Point, np.array([0.45, -0.2, 0.08]))

        req = DualGripperTrajectoryRequest()
        req.gripper1_points.append(target_gripper1_point)
        req.gripper2_points.append(target_gripper2_point)
        action_srv(req)


if __name__ == '__main__':
    main()
