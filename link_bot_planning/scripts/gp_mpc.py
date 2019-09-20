#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import os
import time
from builtins import input

import gpflow as gpf
import numpy as np
import ompl.util as ou
import rospy
import tensorflow as tf
from colorama import Fore
from std_msgs.msg import String

from gazebo_msgs.msg import ContactsState
from ignition import markers
from link_bot_planning import gp_rrt
from link_bot_gaussian_process import link_bot_gp
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest, LinkBotState, LinkBotStateRequest
from link_bot_classifiers.sdf_function_model import SDFFunctionModelRunner
from link_bot_pycommon import link_bot_pycommon
from link_bot_pycommon.link_bot_sdf_utils import SDF

dt = 0.1
success_dist = 0.10
in_contact = False


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


def common(args, start, max_steps=1e6):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.01))
    gpf.reset_default_session(config=config)

    max_v = 0.3
    sdf_data = SDF.load(args.sdf)

    args_dict = vars(args)
    args_dict['random_init'] = False
    fwd_gp_model = link_bot_gp.LinkBotGP()
    fwd_gp_model.load(os.path.join(args.gp_model_dir, 'fwd_model'))
    inv_gp_model = link_bot_gp.LinkBotGP()
    inv_gp_model.load(os.path.join(args.gp_model_dir, 'inv_model'))

    ##############################################
    #             NN Constraint Model            #
    ##############################################
    constraint_tf_graph = tf.Graph()
    with constraint_tf_graph.as_default():
        model = SDFFunctionModelRunner.load(args.checkpoint)
        model = model.change_sdf_shape(sdf_data.sdf.shape[0], sdf_data.sdf.shape[1])
        model.initialize_auxiliary_models()

        class ConstraintCheckerWrapper:

            def __init__(self):
                pass

            @staticmethod
            def get_graph():
                return constraint_tf_graph

            def __call__(self, np_state):
                predicted_violated, predicted_point = model.violated(np_state, sdf_data)
                return predicted_violated

    #############################################
    #           R_k Constraint Model            #
    #############################################
    # def sdf_violated(np_state):
    #     R_k = np.array([[0, 0],
    #                     [0, 0],
    #                     [0, 0],
    #                     [0, 0],
    #                     [1, 0],
    #                     [0, 1]])
    #     pt = np_state @ R_k
    #     x = pt[0, 0]
    #     y = pt[0, 1]
    #     row_col = link_bot_sdf_utils.point_to_sdf_idx(x, y, sdf_resolution, sdf_origin)
    #     violated = sdf[row_col] < 0.02
    #     return violated

    rrt = gp_rrt.GPRRT(fwd_gp_model, inv_gp_model, ConstraintCheckerWrapper(), dt, max_v, args.planner_timeout)

    get_state = rospy.ServiceProxy('/link_bot_state', LinkBotState)

    rospy.init_node('MPCAgent')

    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    action_mode = rospy.Publisher('/link_bot_action_mode', String, queue_size=1, latch=True)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    action_pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=10)
    action_mode_pub = rospy.Publisher("/link_bot_action_mode", String, queue_size=1)
    rospy.Subscriber("/head_contact", ContactsState, contacts_callback)

    state_req = LinkBotStateRequest()
    action_msg = LinkBotVelocityAction()

    action_mode_msg = String()
    action_mode_msg.data = "velocity"
    action_mode_pub.publish(action_mode_msg)

    # Statistics
    num_fails = 0
    num_successes = 0
    execution_times = []
    planning_times = []
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
            config_pub.publish(config)
            time.sleep(0.1)

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
            start_time = time.time()
            while step_idx < max_steps and not done:
                s = get_state(state_req)
                s = np.array([[pt.x, pt.y] for pt in s.points]).reshape(1, -1)
                planned_actions, _, planning_time = rrt.plan(s, goal, sdf_data.sdf, args.verbose)

                if planned_actions is None:
                    num_fails += 1
                    break

                for i, action in enumerate(planned_actions):
                    if i >= args.num_actions > 0:
                        break

                    # publish the pull command
                    action_msg.gripper1_velocity.x = action[0]
                    action_msg.gripper1_velocity.y = action[1]
                    action_pub.publish(action_msg)

                    step = WorldControlRequest()
                    step.steps = int(dt / 0.001)  # assuming 0.001s of simulation time per step
                    world_control.call(step)  # this will block until stepping is complete

                    # check if we are now in collision
                    if in_contact:
                        contacts += 1

                    links_state = get_state(state_req)
                    s_next = np.array([[pt.x, pt.y] for pt in links_state.points]).reshape(1, -1)
                    true_cost = link_bot_pycommon.state_cost(s_next, goal)

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
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            planning_times.append(planning_time)
            min_true_costs.append(min_true_cost)
            nums_contacts.append(contacts)
            nums_steps.append(step_idx)

    except rospy.service.ServiceException:
        pass
    except KeyboardInterrupt:
        pass

    return np.array(min_true_costs), np.array(execution_times), np.array(planning_times), np.array(
        nums_contacts), num_fails, num_successes


def test(args):
    start = np.array([[2, -4, 0, 0, 0, 0]])
    args.n_trials = 1
    common(args, start, max_steps=args.max_steps)


def eval(args):
    stats_filename = os.path.join(args.gp_model_dir, 'eval_{}.txt'.format(int(time.time())))
    start = np.array([[-1, 0, 0, 0, 0, 0]])

    min_costs, execution_times, planning_times, nums_contacts, num_fails, num_successes = common(args, start, max_steps=1)

    eval_stats_lines = [
        '% fail: {}'.format(float(num_fails) / args.n_trials),
        '% success: {}'.format(float(num_successes) / args.n_trials),
        'mean min dist to goal: {}'.format(np.mean(min_costs)),
        'std min dist to goal: {}'.format(np.std(min_costs)),
        'mean planning time: {}'.format(np.mean(planning_times)),
        'std planning time: {}'.format(np.std(planning_times)),
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
    parser.add_argument("checkpoint", help="model checkpoint")
    parser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    parser.add_argument("--model-name", '-m', default="link_bot")
    parser.add_argument("--seed", '-s', type=int, default=1)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--plot-plan", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num-actions", '-T', help="number of actions to execute from the plan", type=int, default=-1)
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=60.0)
    parser.add_argument("--n-trials", '-n', type=int, default=20)

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
