#!/usr/bin/env python
from __future__ import print_function

import argparse
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import rospy
from builtins import input
from link_bot_agent import gurobi_act
from link_bot_gazebo.msg import LinkBotConfiguration
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from link_bot_notebooks import linear_tf_model
from link_bot_notebooks import toy_problem_optimization_common as tpo
from sensor_msgs.msg import Joy

from agent import GazeboAgent

dt = 0.1
success_dist = 0.1


def common(args, goals, max_steps=1e6, verbose=False):
    model = linear_tf_model.LinearTFModel(vars(args), N=args.N, M=args.M, L=args.L, n_steps=args.n_steps, dt=dt)
    agent = GazeboAgent(N=args.N, M=args.M, dt=dt, model=model, gazebo_model_name=args.model_name)

    rospy.init_node('MPCAgent')

    np.random.seed(0)
    joy_msg = Joy()
    joy_msg.axes = [0, 0]

    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)

    # load our initial model
    model.load()

    max_v = 1
    min_true_costs = []

    try:
        for goal in goals:
            # reset to random starting point
            config = LinkBotConfiguration()
            config.tail_pose.x = np.random.uniform(-5, 5)
            config.tail_pose.y = np.random.uniform(-5, 5)
            config.tail_pose.theta = np.random.uniform(-np.pi, np.pi)
            config.joint_angles_rad = np.random.uniform(-np.pi, np.pi, size=2)
            config_pub.publish(config)
            if verbose:
                print("goal: {}".format(np.array2string(goal)))
            og = model.reduce(goal)
            action_selector = gurobi_act.GurobiAct(model, og, max_v)
            min_true_cost = 1e9
            step_idx = 0
            done = False
            while step_idx < max_steps and not done:
                s = agent.get_state()
                o = model.reduce(s)
                actions = action_selector.act(o)

                if np.linalg.norm(actions[0]) < 0.1:
                    done = True

                current_cost = model.cost(o, goal)
                next_cost = model.predict_cost(o, actions, goal)
                if verbose and next_cost > current_cost:
                    print("Local Minimum in cost {} < {}".format(current_cost, next_cost))

                if args.plot_plan:
                    plt.figure()
                    if verbose:
                        print(actions)
                    plt.plot(actions[:, 0, 0], label="x velocity")
                    plt.plot(actions[:, 1, 0], label="y velocity")
                    plt.xlabel("time steps")
                    plt.ylabel("velocity (m/s)")
                    plt.legend()
                    plt.show()
                    return

                for i, action in enumerate(actions):
                    joy_msg.axes = [-action[0, 0], action[1, 0]]

                    joy_pub.publish(joy_msg)
                    step = WorldControlRequest()
                    step.steps = dt / 0.001  # assuming 0.001s of simulation time per step
                    world_control.call(step)  # this will block until stepping is complete

                    s = agent.get_state()
                    true_cost = agent.state_cost(s, goal)

                    if args.pause:
                        input('paused...')

                    if verbose:
                        print("{:0.3f}".format(true_cost))
                    min_true_cost = min(min_true_cost, true_cost)
                    if true_cost < success_dist:
                        if verbose:
                            print("Success!")
                        done = True
                        break
                    step_idx += 1

                if done:
                    break
            min_true_costs.append(min_true_cost)
    except rospy.service.ServiceException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        joy_msg.axes = [0, 0]
        joy_pub.publish(joy_msg)
        if verbose:
            print()
    if verbose:
        print("Min true cost: {}".format(min_true_cost))
    return np.array(min_true_costs)


def test(args):
    goal = np.array([[0], [0], [0], [1], [0], [2]])
    common(args, [goal], verbose=args.verbose)


def eval(args):
    fname = os.path.join(os.path.dirname(args.checkpoint), 'eval_{}.txt'.format(int(time.time())))
    g0 = np.array([[0], [0], [0], [1], [0], [2]])
    goals = [g0] * args.n_random_goals
    min_costs = common(args, goals, max_steps=300)
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
    parser.add_argument("--model-name", '-m', default="myfirst")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--plot-plan", action='store_true')
    parser.add_argument("--tf-debug", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("--n-steps", "-s", type=int, default=1)

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
