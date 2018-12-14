#!/usr/bin/env python
from __future__ import print_function

import argparse
from builtins import input
import numpy as np
import matplotlib.pyplot as plt
from link_bot_gazebo.srv import WorldControl, WorldControlRequest
from sensor_msgs.msg import Joy
import rospy

from link_bot_notebooks.linear_tf_model import LinearTFModel
from agent import GazeboAgent
from link_bot_agent import gurobi_act


def h(n1, n2):
    return np.linalg.norm(np.array(n1) - np.array(n2))


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

    args = parser.parse_args()
    args = args
    dt = 0.1
    model = LinearTFModel(vars(args), N=args.N, M=args.M, L=args.L, n_steps=args.n_steps, dt=dt)
    agent = GazeboAgent(N=args.N, M=args.M, dt=dt, model=model, gazebo_model_name=args.model_name)

    rospy.init_node('MPCAgent')

    np.random.seed(0)
    joy_msg = Joy()
    joy_msg.axes = [0, 0]

    goal = np.array([[0], [0], [0], [1], [0], [2]])

    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)

    # load our initial model
    model.load()

    og = model.reduce(goal)
    max_v = 1
    action_selector = gurobi_act.GurobiAct(model, og, max_v)

    # used for most of our planning algorithms
    done = False

    try:
        while not done:
            s = agent.get_state()
            o = model.reduce(s)
            actions = action_selector.act(o)
            # actions, cs, os, ss = agent.greedy_plan(o, goal)
            # actions = agent.a_star_plan(o, og)

            if args.plot_plan:
                plt.figure()
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

                print("{:0.3f}".format(true_cost))
                # print(action.T)
                if true_cost < 0.1:
                    print("Success!")
                    done = True
                    break

            if done:
                break
    except KeyboardInterrupt:
        pass
    finally:
        joy_msg.axes = [0, 0]
        joy_pub.publish(joy_msg)


if __name__ == '__main__':
    main()
