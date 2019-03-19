#!/usr/bin/env python
from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np
import rospy
from builtins import input
from src.link_bot_agent import one_step_gurobi_act
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from src.link_bot_notebooks import experiments_util
from src.link_bot_notebooks import linear_tf_model
from src.link_bot_notebooks import toy_problem_optimization_common as tpo
from sensor_msgs.msg import Joy

from link_bot.link_bot_agent.src.link_bot_agent.agent import GazeboAgent


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="load this saved model file")
    parser.add_argument("dataset", help="dataset used to train the above checkpoint")
    parser.add_argument("--model-name", '-m', default="myfirst")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--plot-plan", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("--n-steps", "-s", type=int, default=1)

    args = parser.parse_args()
    args = args
    dt = 0.1
    # we use n_steps=1 because in the action selector we want to compute
    model = linear_tf_model.LinearTFModel(vars(args), N=args.N, M=args.M, L=args.L, n_steps=1, dt=dt)
    agent = GazeboAgent(N=args.N, M=args.M, dt=dt, model=model, gazebo_model_name=args.model_name)

    rospy.init_node('DaggerMPCAgent')

    np.random.seed(0)
    joy_msg = Joy()
    joy_msg.axes = [0, 0]

    goal = np.array([[0], [0], [0], [1], [0], [2]])

    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)

    # load our initial model
    model.load()
    dataset = tpo.load_train(args.dataset, n_steps=args.n_steps, N=args.N, L=args.L,
                       extract_func=tpo.link_pos_vel_extractor2(args.N))
    log_path = experiments_util.experiment_name(args.log)

    og = model.reduce(goal)
    max_v = 1
    action_selector = one_step_gurobi_act.GurobiAct(model, og, max_v)

    # generate some goals to train on
    training_goals = []
    for r in np.random.randn(args.n_goals, 4):
        x = r[0] * 5
        y = r[1] * 5
        theta1 = r[2] * np.pi / 2
        theta2 = r[3] * np.pi / 2
        x1 = x + np.cos(theta1)
        y1 = y + np.sin(theta1)
        x2 = x1 + np.cos(theta2)
        y2 = y1 + np.sin(theta2)
        training_goal = np.array([[x], [y], [x1], [y1], [x2], [y2]])
        training_goals.append(training_goal)

    # used for most of our planning algorithms
    done = False

    try:
        Q = 200
        new_data = np.ndarray((args.n_steps, 8, Q))
        for j in range(Q):
            s = agent.get_state()
            o = model.reduce(s)
            actions = action_selector.act(o)

            if args.plot_plan:
                plt.figure()
                plt.plot(actions[:, 0, 0], label="x velocity")
                plt.plot(actions[:, 1, 0], label="y velocity")
                plt.xlabel("time steps")
                plt.ylabel("velocity (m/s)")
                plt.legend()
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

                print("{:0.3f}".format(true_cost))
                if true_cost < 0.1:
                    print("Success!")
                    done = True
                    break

            if done:
                break

        # aggregate
        dataset.append(new_data)
        for training_goal in training_goals:
            model.train(dataset, training_goal, 100, log_path)
    except KeyboardInterrupt:
        pass
    finally:
        joy_msg.axes = [0, 0]
        joy_pub.publish(joy_msg)


if __name__ == '__main__':
    main()
