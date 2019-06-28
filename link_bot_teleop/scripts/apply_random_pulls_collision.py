#!/usr/bin/env python
import argparse
from time import sleep

import numpy as np
import rospy
from gazebo_msgs.srv import GetLinkState
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from link_bot_agent import agent
from link_bot_pycommon import link_bot_pycommon


class LinkConfig:

    def __init__(self):
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None


def main():
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

    # get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
    get_state = rospy.ServiceProxy('/link_bot_state', tLinkState)
    action_pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=10)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    world_control = rospy.ServiceProxy('/world_control', WorldControl)

    # teleport to the origin
    init_config = LinkBotConfiguration()
    init_config.tail_pose.x = -1
    init_config.tail_pose.y = 0
    init_config.tail_pose.theta = 0
    init_config.joint_angles_rad = [0, 0]
    config_pub.publish(init_config)

    # let the world step once
    step = WorldControlRequest()
    step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
    world_control.call(step)  # this will block until stepping is complete

    sleep(0.5)

    def r():
        return np.random.uniform(-np.pi, np.pi)

    times = np.ndarray((args.pulls, args.steps + 1, 1))
    states = np.ndarray((args.pulls, args.steps + 1, args.N))
    actions = np.ndarray((args.pulls, args.steps, args.L))
    constraints = np.ndarray((args.pulls, args.steps + 1, args.Q))
    np.random.seed(args.seed)
    action_msg = LinkBotVelocityAction()
    action_msg.control_link_name = 'head'
    for p in range(0, args.pulls):
        if args.verbose:
            print('=' * 180)

        v = np.random.uniform(0.0, 1.0)
        pull_yaw = r()
        head_vx = np.cos(pull_yaw) * v
        head_vy = np.sin(pull_yaw) * v

        time = 0
        t = 1
        for t in range(args.steps):
            # save the state and action data
            links_state = agent.get_state(get_link_state)
            times[p, t] = [time]
            states[p, t] = links_state
            actions[p, t] = [head_vx, head_vy]

            constraints[p, t, 0] = in_collision
            constraints[p, t, 1] = overstretched

            # publish the pull command
            action_msg.vx = head_vx
            action_msg.vy = head_vy
            action_pub.publish(action_msg)

            # let the simulator run
            step = WorldControlRequest()
            step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
            world_control.call(step)  # this will block until stepping is complete

            # now get the last applied force and whether the object is in collision

            time += DT

        # save the final state
        t += 1
        links_state = agent.get_state(get_link_state)
        times[p, t] = [time]
        states[p, t] = links_state
        distance_to_obstacle_constraint = 0.20
        constraints[p, t, 0] = ?
        constraints[p, t, 1] = ?

        if p % args.save_frequency == 0:
            np.savez(args.outfile,
                     times=times,
                     states=states,
                     actions=actions,
                     constraints=constraints)
            print(p, 'saving data...')

    np.savez(args.outfile,
             times=times,
             states=states,
             actions=actions,
             constraints=constraints)
    print(p, 'saving data...')


if __name__ == '__main__':
    main()
