#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import sleep

import numpy as np
import rospy
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.msg import ContactsState
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotAction
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
        if state.collision1_name == "link_bot::head::head_collision" \
                and state.collision2_name != "ground_plane::link::collision":
            in_contact = True
        if state.collision2_name == "link_bot::head::head_collision" \
                and state.collision1_name != "ground_plane::link::collision":
            in_contact = True


def main():
    global in_contact
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help='filename to store data in')
    parser.add_argument("pulls", help='how many pulls to do', type=int)
    parser.add_argument("steps", help='how many time steps per pull', type=int)
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("-Q", help="dimensions in constraint checking output space", type=int, default=1)
    parser.add_argument("--save-frequency", '-f', help='save every this many steps', type=int, default=10)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    args = args
    rospy.init_node('apply_random_pulls')

    DT = 0.1  # seconds per time step

    joy_msg = Joy()
    joy_msg.axes = [0, 0]
    get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    action_pub = rospy.Publisher("/link_bot_action", LinkBotAction, queue_size=10)
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

    times = np.ndarray((args.pulls, args.steps + 1, 1))
    states = np.ndarray((args.pulls, args.steps + 1, args.N))
    actions = np.ndarray((args.pulls, args.steps, args.L))
    constraints = np.ndarray((args.pulls, args.steps + 1, args.Q))
    np.random.seed(args.seed)
    action_msg = LinkBotAction()
    for p in range(1, args.pulls + 1):
        if args.verbose:
            print('=' * 180)

        v = np.random.uniform(0.0, 1.0)
        pull_yaw = r()
        head_vx = np.cos(pull_yaw) * v
        head_vy = np.sin(pull_yaw) * v

        time = 0
        for t in range(args.steps):
            # save the state and action data
            links_state = agent.get_state(get_link_state)
            times[p, t] = [time]
            states[p, t] = links_state
            actions[p, t] = [head_vx, head_vy]
            constraints[p, t] = [in_contact]

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

        # save the final state
        links_state = agent.get_state(get_link_state)
        times[p, t] = [time]
        states[p, t] = links_state
        constraints[p, t] = [in_contact]

        if p % args.save_frequency == 0:
            np.savez(args.outfile,
                     times=times,
                     states=states,
                     actions=actions,
                     constraints=constraints)
            print(p, 'saving data...')


if __name__ == '__main__':
    main()
