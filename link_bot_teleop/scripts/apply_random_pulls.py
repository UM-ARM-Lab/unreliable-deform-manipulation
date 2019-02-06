#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import sleep

import numpy as np
import rospy
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest
from link_bot_gazebo.msg import LinkBotConfiguration
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from sensor_msgs.msg import Joy


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help='filename to store data in')
    parser.add_argument("pulls", help='how many pulls to do', type=int)
    parser.add_argument("steps", help='how many time steps per pull', type=int)
    parser.add_argument("--save-frequency", '-f', help='save every this many steps', type=int, default=50)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    args = args
    rospy.init_node('apply_random_pulls')

    DT = 0.1  # seconds per time step
    time = 0

    joy_msg = Joy()
    joy_msg.axes = [0, 0]
    joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    world_control = rospy.ServiceProxy('/world_control', WorldControl)

    while config_pub.get_num_connections() < 1:
        sleep(1)

    link_names = ['link_0', 'link_1', 'head']
    S = 4 * len(link_names)

    def get_state(vx, vy):
        # get the new states
        state = np.zeros(S + 1)
        state[0] = time
        for i, link_name in enumerate(link_names):
            req = GetLinkStateRequest()
            req.link_name = link_name
            response = get_link_state.call(req)
            state[1 + 4 * i] = response.link_state.pose.position.x
            state[1 + 4 * i + 1] = response.link_state.pose.position.y
        state[1 + S - 2] = vx
        state[1 + S - 1] = vy

        return state

    def r():
        return np.random.uniform(-np.pi, np.pi)

    joy_msg.axes = [0, 0]
    joy_pub.publish(joy_msg)

    data = []
    np.random.seed(0)
    for p in range(args.pulls):
        if args.verbose:
            print('=' * 180)

        v = np.random.uniform(0.0, 1.0)
        pull_yaw = r()
        vx = np.cos(pull_yaw) * v
        vy = np.sin(pull_yaw) * v

        # set the configuration of the model
        config = LinkBotConfiguration()
        config.tail_pose.x = np.random.uniform(-5, 5)
        config.tail_pose.y = np.random.uniform(-5, 5)
        config.tail_pose.theta = r()
        # allow the rope to be bent
        config.joint_angles_rad = [r() * 0.9, 0]
        config_pub.publish(config)
        time = 0

        for t in range(args.steps + 1):
            # save the state and action data
            data.append(get_state(vx, vy))

            # publish the pull command
            joy_msg.axes = [-vx, vy]  # stupid xbox controller
            joy_pub.publish(joy_msg)

            # let the simulator run
            step = WorldControlRequest()
            step.steps = DT / 0.001  # assuming 0.001s per simulation step
            world_control.call(step)  # this will block until stepping is complete

            time += DT

            if args.verbose:
                print(data[-1])

        if p % args.save_frequency == 0:
            np.save(args.outfile, data)

    # stop everything
    joy_msg.axes = [0, 0]
    joy_pub.publish(joy_msg)


if __name__ == '__main__':
    main()
