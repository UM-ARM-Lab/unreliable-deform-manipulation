#!/usr/bin/env python

import os
import time
import rospy
import argparse
import actionlib
import rosbag

from control_msgs.msg import FollowJointTrajectoryAction
from arm_video_recorder.srv import TriggerVideoRecording, TriggerVideoRecordingRequest


def main():
    rospy.init_node('follow_joint_trajectory_client')

    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile")
    parser.add_argument("--no-video", action='store_true')

    args = parser.parse_args()

    record = rospy.ServiceProxy('video_recorder', TriggerVideoRecording)

    start_msg = TriggerVideoRecordingRequest()
    parent_dir = os.path.join(*os.path.split(args.bagfile)[:-1])
    filename = os.path.join(parent_dir, "trajectory_playback-" + str(int(time.time())) + '.avi')
    if not args.no_video:
        start_msg.record = True
        start_msg.filename = filename
        start_msg.timeout_in_sec = 600.0
        record(start_msg)

    client = actionlib.SimpleActionClient('both_arms_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    client.wait_for_server()

    bag = rosbag.Bag(args.bagfile)
    for _, msg, _ in bag.read_messages():
        goal = msg.goal
        goal.trajectory.header.stamp = rospy.Time.now()
        client.send_goal(goal)
        client.wait_for_result(rospy.Duration.from_sec(600.0))
        result = client.get_result()
        if client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            print("Motion failed:")
            print(client.get_goal_status_text())
            print("Aborting")
            return

    bag.close()

    if not args.no_video:
        stop_msg = TriggerVideoRecordingRequest()
        stop_msg.record = False
        record(stop_msg)


if __name__ == '__main__':
    main()
