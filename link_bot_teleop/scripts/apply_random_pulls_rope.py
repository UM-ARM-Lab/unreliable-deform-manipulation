#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import sleep

import numpy as np
import rospy
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from link_bot_agent import agent


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help='filename to store data in')
    parser.add_argument("pulls", help='how many pulls to do', type=int)
    parser.add_argument("steps", help='how many time steps per pull', type=int)
    parser.add_argument("num_links", help='number of links in the rope', type=int, default=20)
    parser.add_argument("--save-frequency", '-f', help='save every this many steps', type=int, default=10)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    args = args
    rospy.init_node('apply_random_pulls')

    DT = 0.1  # seconds per time step

    action_msg = LinkBotAction()
    action_pub = rospy.Publisher("/link_bot_action", LinkBotAction, queue_size=10)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    world_control = rospy.ServiceProxy('/world_control', WorldControl)

    print("waiting.", end='')
    while config_pub.get_num_connections() < 1:
        print('.', end='')
        sleep(1)

    print("ready...")

    link_names = ['link_{}'.format(i) for i in range(args.num_links)]
    S = 4 * len(link_names)

    def r():
        return np.random.uniform(-np.pi, np.pi)

    # TODO: zero all forces?

    data = []
    np.random.seed(args.seed)
    for p in range(1, args.pulls + 1):
        if args.verbose:
            print('=' * 180)

        f = np.random.uniform(0.0, 100.0)
        control_link_i = np.random.randint(0, len(link_names))
        control_link_name = link_names[control_link_i]
        pull_yaw = r()
        fx = np.cos(pull_yaw) * f
        fy = np.sin(pull_yaw) * f

        # set the configuration of the model
        config = LinkBotConfiguration()
        config.tail_pose.x = np.random.uniform(-3, 3)
        config.tail_pose.y = np.random.uniform(-3, 3)
        config.joint_angles_rad = []
        for i in range(args.num_links):
            config.tail_pose.theta = r()
            # allow the rope to be slightly bent
            config.joint_angles_rad.append(r() * 0.4)
        config_pub.publish(config)
        time = 0

        traj = []
        for t in range(args.steps + 1):
            # save the state and action data
            traj.append(agent.get_rope_data(get_link_state, args.num_links, control_link_i, fx, fy))

            # publish the pull command
            action_msg.control_link_name = control_link_name
            action_msg.use_force = True
            action_msg.wrench.force.x = fx
            action_msg.wrench.force.y = fy
            action_pub.publish(action_msg)

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

    # TODO: zero all forces?


if __name__ == '__main__':
    main()
