#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import sleep

import numpy as np
import rospy
from geometry_msgs.msg import Wrench
from gazebo_msgs.srv import GetLinkState
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotForceAction
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

    DT = 0.01  # seconds per time step

    action_pub = rospy.Publisher("/link_bot_force_action", LinkBotForceAction, queue_size=10)
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

        # set the configuration of the model
        config = LinkBotConfiguration()
        config.tail_pose.x = np.random.uniform(-3, 3)
        config.tail_pose.y = np.random.uniform(-3, 3)
        config.joint_angles_rad = []
        for i in range(args.num_links):
            config.tail_pose.theta = r()
            # allow the rope to be slightly bent
            config.joint_angles_rad.append(r() * 0.2)
        config_pub.publish(config)

        # wait for this to take effect
        sleep(0.5)

        # pick slopes for varying force on each line
        slopes = np.random.uniform(-3, 3, (args.num_links, 2))
        initial_forces = np.random.uniform(-15, 15, (args.num_links, 2))

        time = 0
        traj = []
        for t in range(args.steps + 1):
            # publish the pull command
            current_forces = initial_forces + time * slopes
            action_msg = LinkBotForceAction()
            for i in range(args.num_links):
                wrench = Wrench()
                wrench.force.x = current_forces[i, 0]
                wrench.force.y = current_forces[i, 1]
                action_msg.wrenches.append(wrench)
            # send the command, but this won't have any effect until we call step
            action_pub.publish(action_msg)

            # save the state and action data
            rope_data = agent.get_rope_data(get_link_state, args.num_links)
            rope_data[2] = current_forces
            traj.append(rope_data)

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
