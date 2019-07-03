#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import sleep

import numpy as np
import rospy
import matplotlib.pyplot as plt
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.msg import ContactsState
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
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
        if state.collision1_name == "link_bot::head::head_collision" \
                and state.collision2_name != "ground_plane::link::collision":
            in_contact = True
        if state.collision2_name == "link_bot::head::head_collision" \
                and state.collision1_name != "ground_plane::link::collision":
            in_contact = True


def generate(args):
    rospy.init_node('test_pull')

    DT = 0.1  # seconds per time step

    get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    action_pub = rospy.Publisher("/link_bot_action", LinkBotAction, queue_size=10)
    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    rospy.Subscriber("/head_contact", ContactsState, contacts_callback)

    # Set initial position
    init_config = LinkBotConfiguration()
    init_config.tail_pose.x = -1
    init_config.tail_pose.y = 0
    init_config.tail_pose.theta = 0
    init_config.joint_angles_rad = [0, 0]
    config_pub.publish(init_config)

    data = []
    head_vx = 1
    head_vy = 0

    sleep(0.5)

    time = 0
    traj = []
    action_msg = LinkBotAction()
    for t in range(args.steps + 1):
        # save the state and action data
        data_point = agent.get_time_state_action_collision(get_link_state, time, head_vx, head_vy, in_contact)
        traj.append(data_point)

        # publish the pull command
        action_msg.control_link_name = 'head'
        action_msg.use_force = False
        action_msg.twist.linear.x = head_vx
        action_msg.twist.linear.y = head_vy
        action_pub.publish(action_msg)

        # let the simulator run
        step = WorldControlRequest()
        step.steps = DT / 0.001  # assuming 0.001s per simulation step
        world_control.call(step)  # this will block until stepping is complete

        time += DT

        if args.verbose:
            print(data[-1])

    data.append(traj)

    np.save(args.outfile, data)


def plot(args):
    data = np.load(args.infile)
    data = data.squeeze()

    print(data)

    plt.figure()
    plt.title("collision bit")
    plt.plot(data[:, 1])

    plt.figure()
    plt.plot(data[:, 8], label='vx')
    plt.plot(data[:, 9], label='vy')
    plt.title("control input")
    plt.xlabel("time")
    plt.ylabel("velocity (m/s)")

    plt.figure()
    plt.plot(data[:, 2], data[:, 3], label='link_0')
    plt.plot(data[:, 4], data[:, 5], label='link_1')
    plt.plot(data[:, 6], data[:, 7], label='head')
    plt.title("link positions")
    plt.legend()
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


def main():
    global in_contact
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    generate_subparser = subparsers.add_parser("generate")

    generate_subparser.add_argument("outfile", help='filename to store data in')
    generate_subparser.add_argument("steps", help='how many time steps per pull', type=int)
    generate_subparser.add_argument("--save-frequency", '-f', help='save every this many steps', type=int, default=10)
    generate_subparser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    generate_subparser.add_argument("--verbose", '-v', action="store_true")
    generate_subparser.set_defaults(func=generate)

    plot_subparser = subparsers.add_parser("plot")
    plot_subparser.add_argument("infile")
    plot_subparser.set_defaults(func=plot)

    args = parser.parse_args()

    args = args
    args.func(args)


if __name__ == '__main__':
    main()
