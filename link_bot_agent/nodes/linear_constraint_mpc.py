#!/usr/bin/env python
from __future__ import print_function

import argparse
import ompl.util as ou
import os
import time as timemod

import numpy as np
import rospy
from builtins import input
from colorama import Fore
from gazebo_msgs.msg import ContactsState
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest

from ignition import markers
from link_bot_agent import agent, ompl_act, one_step_action_selector, lqr_action_selector, \
    dual_lqr_action_selector
from link_bot_agent.gurobi_directed_control_sampler import GurobiDirectedControlSampler
from link_bot_agent.lqr_directed_control_sampler import LQRDirectedControlSampler
from link_bot_notebooks import linear_constraint_model, linear_tf_model
from link_bot_notebooks import toy_problem_optimization_common as tpoc

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
    global in_contact
    if args.logdir:
        now = int(timemod.time())
        os.path.split(args.checkpoint)
        checkpoint_path = os.path.normpath(args.checkpoint)
        folders = checkpoint_path.split(os.sep)
        checkpoint_folders = []
        relevant = False
        for folder in folders:
            if relevant:
                checkpoint_folders.append(folder)
            if folder == "log_data":
                relevant = True

        logfile = os.path.join(args.logdir, "{}_{}.npz".format("_".join(checkpoint_folders), now))
        print(Fore.CYAN + "Saving new data in {}".format(logfile) + Fore.RESET)

    batch_size = 1
    max_v = 1
    n_steps = 1
    n_steps_for_logging = 50
    if args.controller == 'ompl-dual-lqr':
        sdf, sdf_gradient, sdf_resolution = tpoc.load_sdf(args.sdf)
        sdf_rows, sdf_cols = sdf.shape
        sdf_origin_coordinate = np.array([sdf_rows / 2, sdf_cols / 2], dtype=np.int32)

        def sdf_by_xy(x, y):
            point = np.array([[x, y]])
            indeces = (point / sdf_resolution).astype(np.int32) + sdf_origin_coordinate
            return sdf[indeces[0, 0], indeces[0, 1]]

        args_dict = vars(args)
        args_dict['random_init'] = False
        tf_model = linear_constraint_model.LinearConstraintModel(args_dict, sdf, sdf_gradient, sdf_resolution,
                                                                 batch_size, args.N, args.M, args.L, args.P, args.Q, dt,
                                                                 n_steps)
        tf_model.load()
        lqr_solver = dual_lqr_action_selector.DualLQRActionSelector(tf_model, max_v)
        action_selector = ompl_act.OMPLAct(tf_model, lqr_solver, LQRDirectedControlSampler, dt, max_v,
                                           args.planner_timeout)
    else:
        tf_model = linear_tf_model.LinearTFModel(vars(args), batch_size, args.N, args.M, args.L, dt, n_steps)
        tf_model.load()
        sdf = None
        if args.controller == 'ompl-lqr':
            lqr_solver = lqr_action_selector.LQRActionSelector(tf_model, max_v)
            action_selector = ompl_act.OMPLAct(tf_model, lqr_solver, LQRDirectedControlSampler, dt, max_v,
                                               args.planner_timeout)
        if args.controller == 'ompl-gurobi':
            gurobi_solver = one_step_action_selector.OneStepGurobiAct(tf_model, max_v)
            action_selector = ompl_act.OMPLAct(tf_model, gurobi_solver, GurobiDirectedControlSampler, dt, max_v,
                                               args.planner_timeout)
        elif args.controller == 'gurobi':
            action_selector = one_step_action_selector.OneStepGurobiAct(tf_model, max_v)

    gzagent = agent.GazeboAgent(N=args.N, M=args.M, dt=dt, model=tf_model, gazebo_model_name=args.model_name)

    rospy.init_node('MPCAgent')

    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    action_pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=10)
    rospy.Subscriber("/head_contact", ContactsState, contacts_callback)

    times = []
    states = []
    actions = []
    constraints = []
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
            goal = np.zeros((1, 6))
            goal[0, 0] = np.random.uniform(-4.0, 3.0)
            goal[0, 1] = np.random.uniform(-4.0, 4.0)

            config = LinkBotConfiguration()
            config.tail_pose.x = start[0, 0]
            config.tail_pose.y = start[0, 1]
            config.tail_pose.theta = np.random.uniform(-0.2, 0.2)
            config.joint_angles_rad = [np.random.uniform(-0.2, 0.2), 0]
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

            o_d_goal, _ = tf_model.reduce(goal)

            min_true_cost = 1e9
            step_idx = 0
            logging_idx = 0
            done = False
            time_traj = np.ndarray((n_steps_for_logging + 1, 1))
            state_traj = np.ndarray((n_steps_for_logging + 1, args.N))
            action_traj = np.ndarray((n_steps_for_logging, 2))
            constraint_traj = np.ndarray((n_steps_for_logging + 1, args.Q))
            discrete_time = 0
            contacts = 0
            start_time = timemod.time()
            while step_idx < max_steps and not done:
                s = agent.get_state(gzagent.get_link_state)
                o_d, o_k = tf_model.reduce(s)
                planned_actions, _ = action_selector.act(sdf, o_d, o_k, o_d_goal, args.verbose)

                if planned_actions is None:
                    num_fails += 1
                    break

                for i, action in enumerate(planned_actions):
                    if i >= args.num_actions > 0:
                        break

                    links_state = agent.get_state(gzagent.get_link_state)
                    time_traj[logging_idx] = [discrete_time]
                    state_traj[logging_idx] = links_state
                    action_traj[logging_idx] = action[0]
                    sdf_at_head = sdf_by_xy(links_state[4], links_state[5])
                    distance_to_obstacle_constraint = 0.20
                    constraint_traj[logging_idx] = [float(sdf_at_head < distance_to_obstacle_constraint)]

                    # publish the pull command

                    # actions should be scaled in LQR? do it again here?
                    u_norm = np.linalg.norm(action)
                    if u_norm > 1e-9:
                        if u_norm > max_v:
                            scaling = max_v
                        else:
                            scaling = u_norm
                        action = action * scaling / u_norm

                    action_msg.vx = action[0, 0]
                    action_msg.vy = action[0, 1]
                    action_pub.publish(action_msg)

                    step = WorldControlRequest()
                    step.steps = dt / 0.001  # assuming 0.001s of simulation time per step
                    world_control.call(step)  # this will block until stepping is complete

                    # check if we are now in collision
                    if in_contact:
                        contacts += 1

                    s_next = np.array(links_state).reshape(1, args.N)
                    true_cost = gzagent.state_cost(s_next, goal)

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

                    # once we get a full trajectory, append it to the list of trajectories and save everything
                    # then reset the counter so we start filling the trajectory at the beginning
                    if logging_idx == n_steps_for_logging:
                        links_state = agent.get_state(gzagent.get_link_state)
                        time_traj[logging_idx] = [discrete_time]
                        state_traj[logging_idx] = links_state
                        sdf_at_head = sdf_by_xy(links_state[4], links_state[5])
                        distance_to_obstacle_constraint = 0.20
                        constraint_traj[logging_idx] = [float(sdf_at_head < distance_to_obstacle_constraint)]

                        logging_idx = 0
                        times.append(time_traj)
                        states.append(state_traj)
                        actions.append(action_traj)
                        constraints.append(constraint_traj)
                        if args.logdir:
                            print("saving data...")
                            np.savez(logfile,
                                     times=times,
                                     states=states,
                                     actions=actions,
                                     constraints=constraints)

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
    common(args, start, max_steps=1000)


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
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="load this saved model file")
    parser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    parser.add_argument("--model-name", '-m', default="link_bot")
    parser.add_argument("--seed", '-s', type=int, default=2)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--plot-plan", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state o_d", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("-P", help="dimensions in latent state o_k", type=int, default=2)
    parser.add_argument("-Q", help="dimensions in constraint checking output space", type=int, default=1)
    parser.add_argument("--num-actions", '-T', help="number of actions to execute from the plan", type=int, default=10)
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=1.0)
    parser.add_argument("--logdir", '-d', help='data directory to store logged data in')
    parser.add_argument("--controller", choices=['gurobi', 'ompl-lqr', 'ompl-dual-lqr', 'ompl-gurobi'])
    parser.add_argument("--n-trials", '-n', type=int, default=20)

    subparsers = parser.add_subparsers()
    test_subparser = subparsers.add_parser("test")
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
