#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import time as timemod
from builtins import input

import numpy as np
import ompl.util as ou
import tensorflow as tf
from colorama import Fore

import rospy
from gazebo_msgs.srv import GetLinkState
from ignition import markers
from link_bot_agent import agent, gp_rrt
from link_bot_gaussian_process import link_bot_gp
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from link_bot_notebooks import toy_problem_optimization_common as tpoc
from link_bot_notebooks.constraint_model import ConstraintModel
import gpflow as gpf

dt = 0.1
success_dist = 0.10
in_contact = False


def common(args, start, max_steps=1e6):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.01))
    gpf.reset_default_session(config=config)

    max_v = 1
    sdf, sdf_gradient, sdf_resolution, sdf_origin = tpoc.load_sdf(args.sdf)

    args_dict = vars(args)
    args_dict['random_init'] = False
    fwd_gp_model = link_bot_gp.LinkBotGP()
    fwd_gp_model.load(os.path.join(args.gp_model_dir, 'fwd_model'))
    inv_gp_model = link_bot_gp.LinkBotGP()
    inv_gp_model.load(os.path.join(args.model_dir, 'inv_model'))

    args_dict = dict(args)
    constraint_model = ConstraintModel(args_dict, )
    constraint_model.

    def sdf_violated(np_state):
        # This should be loaded from the learned constraint model
        R_k = np.array([[0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [1, 0],
                        [0, 1]])
        pt = np_state @ R_k
        x = pt[0, 0]
        y = pt[0, 1]
        # learned tail selection

        row_col = tpoc.point_to_sdf_idx(x, y, sdf_resolution, sdf_origin)
        violated = sdf[row_col] < args.sdf_threshold
        return violated

    rrt = gp_rrt.GPRRT(fwd_gp_model, inv_gp_model, sdf_violated, dt, max_v, args.planner_timeout)

    get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)

    rospy.init_node('MPCAgent')

    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    action_pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=10)

    action_msg = LinkBotVelocityAction()
    action_msg.control_link_name = 'head'

    # Statistics
    num_fails = 0
    num_successes = 0
    execution_times = []
    min_true_costs = []
    nums_steps = []
    nums_contacts = []

    # Visualization
    start_marker = markers.make_marker(rgb=[0, 1, 1], id=1)
    goal_marker = markers.make_marker(rgb=[0, 1, 0], id=2)

    # Catch planning failure exception
    try:
        for trial_idx in range(args.n_trials):
            goal = np.zeros((1, 2))
            goal[0, 0] = np.random.uniform(-4.0, 3.0)
            goal[0, 1] = np.random.uniform(-4.0, 4.0)

            config = LinkBotConfiguration()
            config.tail_pose.x = start[0, 0]
            config.tail_pose.y = start[0, 1]
            config.tail_pose.theta = np.random.uniform(-0.2, 0.2)
            config.joint_angles_rad = [np.random.uniform(-0.2, 0.2), 0]
            # config.tail_pose.x = -2.4194
            # config.tail_pose.y = 0.323
            # config.tail_pose.theta = 1.2766
            # config.joint_angles_rad = [2.54, 0]
            config_pub.publish(config)
            timemod.sleep(0.1)

            # publish markers
            start_marker.pose.position.x = config.tail_pose.x
            start_marker.pose.position.y = config.tail_pose.y
            goal_marker.pose.position.x = goal[0, 0]
            goal_marker.pose.position.y = goal[0, 1]
            markers.publish(goal_marker)
            markers.publish(start_marker)

            if args.verbose:
                print("start: {}, {}".format(config.tail_pose.x, config.tail_pose.y))
                print("goal: {}, {}".format(goal[0, 0], goal[0, 1]))

            min_true_cost = 1e9
            step_idx = 0
            logging_idx = 0
            done = False
            discrete_time = 0
            contacts = 0
            start_time = timemod.time()
            while step_idx < max_steps and not done:
                s = agent.get_state(get_link_state)
                s = np.array(s).reshape((1, fwd_gp_model.n_state))
                planned_actions, _ = rrt.plan(s, goal, sdf, args.verbose)

                if planned_actions is None:
                    num_fails += 1
                    break

                for i, action in enumerate(planned_actions):
                    if i >= args.num_actions > 0:
                        break

                    # publish the pull command
                    action_msg.vx = action[0]
                    action_msg.vy = action[1]
                    action_pub.publish(action_msg)

                    step = WorldControlRequest()
                    step.steps = int(dt / 0.001)  # assuming 0.001s of simulation time per step
                    world_control.call(step)  # this will block until stepping is complete

                    # check if we are now in collision
                    if in_contact:
                        contacts += 1

                    links_state = agent.get_state(get_link_state)
                    s_next = np.array(links_state).reshape(1, fwd_gp_model.n_state)
                    true_cost = tpoc.state_cost(s_next, goal)

                    if args.pause:
                        input('paused...')

                    min_true_cost = min(min_true_cost, true_cost)
                    if true_cost < success_dist:
                        num_successes += 1
                        done = True
                        if args.verbose:
                            print("Success!")
                    step_idx += 1
                    logging_idx += 1
                    discrete_time += dt

                if done:
                    break
            execution_time = timemod.time() - start_time
            execution_times.append(execution_time)
            min_true_costs.append(min_true_cost)
            nums_contacts.append(contacts)
            nums_steps.append(step_idx)

    except rospy.service.ServiceException:
        pass
    except KeyboardInterrupt:
        pass

    return np.array(min_true_costs), np.array(execution_times), np.array(nums_contacts), num_fails, num_successes


def test(args):
    start = np.array([[-1, 0, 0, 0, 0, 0]])
    args.n_trials = 1
    common(args, start, max_steps=args.max_steps)


def eval(args):
    stats_filename = os.path.join(os.path.dirname(args.checkpoint), 'eval_{}.txt'.format(int(timemod.time())))
    start = np.array([[-1, 0, 0, 0, 0, 0]])

    min_costs, execution_times, nums_contacts, num_fails, num_successes = common(args, start, 200)

    eval_stats_lines = [
        '% fail: {}'.format(float(num_fails) / args.n_trials),
        '% success: {}'.format(float(num_successes) / args.n_trials),
        'mean min dist to goal: {}'.format(np.mean(min_costs)),
        'std min dist to goal: {}'.format(np.std(min_costs)),
        'mean execution time: {}'.format(np.mean(execution_times)),
        'std execution time: {}'.format(np.std(execution_times)),
        'mean num contacts: {}'.format(np.mean(nums_contacts)),
        'std num contacts: {}'.format(np.std(nums_contacts)),
        'full data',
        'min costs: {}'.format(np.array2string(min_costs)),
        'execution times: {}'.format(np.array2string(execution_times)),
        'num contacts: {}'.format(np.array2string(nums_contacts)),
        '\n'
    ]

    print(eval_stats_lines)
    stats_file = open(stats_filename, 'w')
    print(Fore.CYAN + "writing evaluation statistics to: {}".format(stats_filename) + Fore.RESET)
    stats_file.writelines("\n".join(eval_stats_lines))


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("gp_model_dir", help="load this saved forward model file")
    parser.add_argument("tf_constraint_model", help="constraint model checkpoint")
    parser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    parser.add_argument("--model-name", '-m', default="link_bot")
    parser.add_argument("--seed", '-s', type=int, default=2)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--plot-plan", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num-actions", '-T', help="number of actions to execute from the plan", type=int, default=-1)
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=90.0)
    parser.add_argument("--controller", choices=['gurobi', 'ompl-lqr', 'ompl-dual-lqr', 'ompl-gurobi'])
    parser.add_argument("--n-trials", '-n', type=int, default=20)
    parser.add_argument("--sdf-threshold", type=float, help='smallest allowed distance to an obstacle', default=0.20)

    subparsers = parser.add_subparsers()
    test_subparser = subparsers.add_parser("test")
    test_subparser.add_argument('--max-steps', type=int, default=10000)
    test_subparser.set_defaults(func=test)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.set_defaults(func=eval)

    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    # ou.setLogLevel(ou.LOG_DEBUG)
    ou.setLogLevel(ou.LOG_ERROR)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
