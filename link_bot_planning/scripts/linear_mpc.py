#!/usr/bin/env python
from __future__ import print_function

import argparse
from colorama import Fore
import time as timemod
import os

import matplotlib.pyplot as plt
import numpy as np
import rospy
import ompl.util as ou
from builtins import input
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from link_bot_classifiers.src.link_bot_models import linear_tf_model
from link_bot_planning import agent, ompl_act, one_step_action_selector, lqr_action_selector
from link_bot_planning.lqr_directed_control_sampler import LQRDirectedControlSampler
from link_bot_planning.gurobi_directed_control_sampler import GurobiDirectedControlSampler

dt = 0.1
success_dist = 0.1


def common(args, goals, max_steps=1e6, verbose=False):
    if args.logdir:
        now = int(timemod.time())
        os.path.split(args.checkpoint)
        checkpoint_path = os.path.normpath(args.checkpoint)
        folders = checkpoint_path.split(os.sep)
        checkpoint_folders = []
        relavent = False
        for folder in folders:
            if relavent:
                checkpoint_folders.append(folder)
            if folder == "log_data":
                relavent = True

        logfile = os.path.join(args.logdir, "{}_{}.npy".format("_".join(checkpoint_folders), now))
        print(Fore.CYAN + "Saving new data in {}".format(logfile) + Fore.RESET)

    batch_size = 1
    n_steps = 1
    tf_model = linear_tf_model.LinearTFModel(vars(args), batch_size, args.N, args.M, args.L, dt, n_steps)
    gzagent = agent.GazeboAgent(N=args.N, M=args.M, dt=dt, model=tf_model, gazebo_model_name=args.model_name)

    rospy.init_node('MPCAgent')

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_WARN)

    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    action_pub = rospy.Publisher("/link_bot_action", LinkBotAction, queue_size=10)

    # load our initial model
    tf_model.load()

    max_v = 1
    min_true_costs = []
    T = -1

    try:
        times = []
        states = []
        actions = []
        constraints = []
        action_msg = LinkBotAction()
        for goal in goals:
            # reset to random starting point
            config = LinkBotConfiguration()
            config.tail_pose.x = np.random.uniform(-3, 3)
            config.tail_pose.y = np.random.uniform(-3, 3)
            config.tail_pose.theta = np.random.uniform(-np.pi, np.pi)
            config.joint_angles_rad = np.random.uniform(-np.pi, np.pi, size=2)
            config_pub.publish(config)
            timemod.sleep(0.1)

            if verbose:
                print("goal: {}".format(np.array2string(goal)))
            og = tf_model.reduce(goal)
            if args.controller == 'ompl-lqr':
                lqr_solver = lqr_action_selector.LQRActionSelector(tf_model, max_v)
                action_selector = ompl_act.OMPLAct(lqr_solver, LQRDirectedControlSampler, args.M,
                                                   args.L, dt, og, max_v)
            if args.controller == 'ompl-gurobi':
                gurobi_solver = one_step_action_selector.OneStepGurobiAct(tf_model, max_v)
                action_selector = ompl_act.OMPLAct(gurobi_solver, GurobiDirectedControlSampler, args.M,
                                                   args.L, dt, og, max_v)
            elif args.controller == 'gurobi':
                action_selector = one_step_action_selector.OneStepGurobiAct(tf_model, max_v)
            elif args.controller == 'lqr':
                action_selector = lqr_action_selector.LQRActionSelector(tf_model, max_v)

            min_true_cost = 1e9
            step_idx = 0
            done = False
            time_traj = []
            state_traj = []
            action_traj = []
            constraint_traj = []
            time = 0
            while step_idx < max_steps and not done:
                s = agent.get_state(gzagent.get_link_state)
                o = tf_model.reduce(s)
                planned_actions, _ = action_selector.act(o, verbose)

                for i, action in enumerate(planned_actions):
                    if i >= T > 0:
                        break

                    u_norm = np.linalg.norm(action)
                    if u_norm > max_v:
                        scaling = max_v
                    else:
                        scaling = u_norm
                    action = action * scaling / u_norm

                    links_state = agent.get_state(gzagent.get_link_state)
                    time_traj.append(time)
                    state_traj.append(links_state)
                    action_traj.append(action[0])
                    constraint_traj.append(None)

                    # publish the pull command
                    action_msg.control_link_name = 'head'
                    action_msg.use_force = False
                    action_msg.twist.linear.x = action[0, 0]
                    action_msg.twist.linear.y = action[0, 1]
                    action_pub.publish(action_msg)

                    step = WorldControlRequest()
                    step.steps = dt / 0.001  # assuming 0.001s of simulation time per step
                    world_control.call(step)  # this will block until stepping is complete

                    # get new state
                    s_next = np.array(agent.get_state(gzagent.get_link_state)).reshape(1, args.N)
                    true_cost = gzagent.state_cost(s_next, goal)

                    if args.verbose:
                        print(s, action, s_next)

                    if args.pause:
                        input('paused...')

                    min_true_cost = min(min_true_cost, true_cost)
                    if true_cost < success_dist:
                        done = True
                        if verbose:
                            print("Success!")
                    step_idx += 1
                    time += dt

                if done:
                    break

            # save the final state
            links_state = agent.get_state(gzagent.get_link_state)
            time_traj.append(time)
            state_traj.append(links_state)
            constraint_traj.append(None)

            min_true_costs.append(min_true_cost)
            times.append(time_traj)
            states.append(state_traj)
            actions.append(action_traj)
            constraints.append(constraint_traj)
            if args.logdir:
                np.savez(args.outfile,
                         times=times,
                         states=states,
                         actions=actions)
    except rospy.service.ServiceException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if verbose:
            print()
    if verbose:
        print("Min true cost: {}".format(min_true_cost))
    return np.array(min_true_costs)


def test(args):
    goal = np.array([[0, 0, 0, 0, 0, 0]])
    common(args, [goal], verbose=args.verbose)


def eval(args):
    fname = os.path.join(os.path.dirname(args.checkpoint), 'eval_{}.txt'.format(int(timemod.time())))
    g0 = np.array([[0, 0, 0, 0, 0, 0]])
    goals = [g0] * args.n_goals
    min_costs = common(args, goals, max_steps=151)
    print(min_costs)
    print('mean dist to goal', np.mean(min_costs))
    print('stdev dist to goal', np.std(min_costs))
    success_percentage = float(np.count_nonzero(np.where(min_costs < success_dist, 1.0, 0.0))) / len(min_costs)
    print('% success', success_percentage)
    np.savetxt(fname, min_costs)
    plt.hist(min_costs)
    plt.show()


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="load this saved model file")
    parser.add_argument("--model-name", '-m', default="link_bot")
    parser.add_argument("--seed", '-s', type=int, default=2)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--plot-plan", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state o_d", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("--logdir", '-d', help='data directory to store logged data in')
    parser.add_argument("--controller", choices=['gurobi', 'lqr', 'ompl-lqr', 'ompl-gurobi'])

    subparsers = parser.add_subparsers()
    test_subparser = subparsers.add_parser("test")
    test_subparser.set_defaults(func=test)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("--n-goals", '-n', type=int, default=100)
    eval_subparser.set_defaults(func=eval)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()