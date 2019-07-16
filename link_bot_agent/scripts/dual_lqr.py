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
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from src.link_bot.link_bot_sdf_tools.src.link_bot_sdf_tools.link_bot_sdf_tools import SDF
from link_bot_models import linear_constraint_model
from link_bot_agent import agent, dual_lqr_action_selector

dt = 0.1
# TODO: make this lower
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
    max_v = 1
    n_steps = 1
    sdf_data = SDF.load(args.sdf)

    tf_model = linear_constraint_model.LinearConstraintModel(vars(args), sdf_data, batch_size, args.N, args.M, args.L, args.P,
                                                             args.Q, dt, n_steps)
    tf_model.load()
    action_selector = dual_lqr_action_selector.DualLQRActionSelector(tf_model, max_v)

    gzagent = agent.GazeboAgent(N=args.N, M=args.M, dt=dt, model=tf_model, gazebo_model_name=args.model_name)

    rospy.init_node('MPCAgent')

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_WARN)

    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    action_pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=10)

    min_true_costs = []
    T = -1

    try:
        times = []
        states = []
        actions = []
        constraints = []
        action_msg = LinkBotVelocityAction()
        for goal in goals:
            # reset to random starting point
            config = LinkBotConfiguration()
            config.tail_pose.x = -4
            config.tail_pose.y = 1
            config.tail_pose.theta = 0
            config.joint_angles_rad = [0, 0]
            config_pub.publish(config)
            timemod.sleep(0.1)

            if verbose:
                print("goal: {}".format(np.array2string(goal)))

            o_d_goal, _ = tf_model.reduce(goal)

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
                o_d, _ = tf_model.reduce(s)
                planned_actions, _ = action_selector.act(o_d, None, o_d_goal, verbose)

                for i, action in enumerate(planned_actions):
                    if i >= T > 0:
                        break

                    u_norm = np.linalg.norm(action)
                    if u_norm > 1e-9:
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
                    action_msg.vx = action[0, 0]
                    action_msg.vy = action[0, 1]
                    action_pub.publish(action_msg)

                    step = WorldControlRequest()
                    step.steps = dt / 0.001  # assuming 0.001s of simulation time per step
                    world_control.call(step)  # this will block until stepping is complete

                    s_next = np.array(links_state).reshape(1, args.N)
                    true_cost = gzagent.state_cost(s_next, goal)

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
    goal = np.array([[2.5, 0, 0, 0, 0, 0]])
    common(args, [goal], verbose=args.verbose)


def eval(args):
    fname = os.path.join(os.path.dirname(args.checkpoint), 'eval_{}.txt'.format(int(timemod.time())))
    g0 = np.array([[2.5, 0, 0, 0, 0, 0]])
    goals = [g0] * args.n_random_goals
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
    parser.add_argument("--logdir", '-d', help='data directory to store logged data in')

    subparsers = parser.add_subparsers()
    test_subparser = subparsers.add_parser("test")
    test_subparser.set_defaults(func=test)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("--n-random-goals", '-n', type=int, default=100)
    eval_subparser.set_defaults(func=eval)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
