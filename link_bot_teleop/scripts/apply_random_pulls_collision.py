#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import sleep

import numpy as np
import rospy
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.msg import ContactsState
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from link_bot_agent import agent
from link_bot_notebooks import toy_problem_optimization_common as tpoc


class LinkConfig:

    def __init__(self):
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("sdf", help='sdf npz file')
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

    get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
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
    sleep(0.5)

    # load the SDF
    sdf, sdf_gradient, sdf_resolution = tpoc.load_sdf(args.sdf)
    sdf_rows, sdf_cols = sdf.shape
    sdf_origin_coordinate = np.array([sdf_rows / 2, sdf_cols / 2], dtype=np.int32)

    def sdf_by_xy(x, y):
        point = np.array([[x, y]])
        indeces = (point / sdf_resolution).astype(np.int32) + sdf_origin_coordinate
        return sdf[indeces[0, 0], indeces[0, 1]]

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
        for t in range(args.steps):
            # save the state and action data
            links_state = agent.get_state(get_link_state)
            times[p, t] = [time]
            states[p, t] = links_state
            actions[p, t] = [head_vx, head_vy]
            # The task-space constraint is that we stay at least 5cm away from obstacles.
            sdf_at_head = sdf_by_xy(links_state[4], links_state[5])
            distance_to_obstacle_constraint = 0.20
            constraints[p, t] = [float(sdf_at_head < distance_to_obstacle_constraint)]

            # publish the pull command
            action_msg.vx = head_vx
            action_msg.vy = head_vy
            action_pub.publish(action_msg)

            # let the simulator run
            step = WorldControlRequest()
            step.steps = DT / 0.001  # assuming 0.001s per simulation step
            world_control.call(step)  # this will block until stepping is complete

            time += DT

        # save the final state
        t += 1
        links_state = agent.get_state(get_link_state)
        times[p, t] = [time]
        states[p, t] = links_state
        sdf_at_head = sdf_by_xy(links_state[4], links_state[5])
        distance_to_obstacle_constraint = 0.20
        constraints[p, t] = [float(sdf_at_head < distance_to_obstacle_constraint)]

        if p % args.save_frequency == 0:
            np.savez(args.outfile,
                     times=times,
                     states=states,
                     actions=actions,
                     constraints=constraints)
            print(p, 'saving data...')


if __name__ == '__main__':
    main()
