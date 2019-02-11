#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import sleep

import numpy as np
import rospy
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest
from gazebo_msgs.msg import ContactsState
from link_bot_gazebo.msg import LinkBotConfiguration
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from sensor_msgs.msg import Joy
from link_bot_agent import agent

in_contact = False


class LinkConfig:

    def __init__(self):
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None


def contacts_callback(contacts):
    global in_contact
    in_contact = False
    for state in contacts.states:
        if state.collision1_name == "myfirst::head::head_collision" and state.collision2_name != "ground_plane::link::collision":
            in_contact = True


def main():
    global in_contact
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help='filename to store data in')
    parser.add_argument("pulls", help='how many pulls to do', type=int)
    parser.add_argument("steps", help='how many time steps per pull', type=int)
    parser.add_argument("--save-frequency", '-f', help='save every this many steps', type=int, default=10)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    args = args
    rospy.init_node('apply_random_pulls')

    DT = 0.1  # seconds per time step

    joy_msg = Joy()
    joy_msg.axes = [0, 0]
    joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)
    get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    rospy.Subscriber("/head_contact", ContactsState, contacts_callback)

    # Set initial position
    init_config = LinkBotConfiguration()
    init_config.tail_pose.x = 0
    init_config.tail_pose.y = 0
    init_config.tail_pose.theta = 0
    init_config.joint_angles_rad = [0, 0]
    config_pub.publish(init_config)

    def r():
        return np.random.uniform(-np.pi, np.pi)

    joy_msg.axes = [0, 0]
    joy_pub.publish(joy_msg)

    data = []
    np.random.seed(args.seed)
    for p in range(1, args.pulls + 1):
        if args.verbose:
            print('=' * 180)

        v = np.random.uniform(0.0, 1.0)
        pull_yaw = r()
        head_vx = np.cos(pull_yaw) * v
        head_vy = np.sin(pull_yaw) * v

        time = 0
        traj = []
        for t in range(args.steps + 1):
            # save the state and action data
            traj.append(agent.get_time_state_action_collision(get_link_state, time, head_vx, head_vy, in_contact))

            # publish the pull command
            joy_msg.axes = [-head_vx, head_vy]  # stupid xbox controller
            joy_pub.publish(joy_msg)

            # let the simulator run
            step = WorldControlRequest()
            step.steps = DT / 0.001  # assuming 0.001s per simulation step
            world_control.call(step)  # this will block until stepping is complete

            time += DT

            if args.verbose:
                print(data[-1])

        data.append(traj)

        if p % args.save_frequency == 0:
            np.save(args.outfile, data)
            print(p, 'saving data...')

    # stop everything
    joy_msg.axes = [0, 0]
    joy_pub.publish(joy_msg)


if __name__ == '__main__':
    main()
