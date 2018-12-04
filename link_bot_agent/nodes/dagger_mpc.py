#!/usr/bin/env python
from time import sleep
import os
from colorama import Fore
import argparse
import numpy as np
from sensor_msgs.msg import Joy
from std_srvs.srv import Empty, EmptyRequest
import rospy

from link_bot_notebooks import notebook_finder
from link_bot_notebooks import toy_problem_optimization_common as tpo
from link_bot_notebooks import linear_tf_model
from link_bot_notebooks import experiments_util
import agent


def h(n1, n2):
    return np.linalg.norm(np.array(n1) - np.array(n2))


class MPCAgent:

    def __init__(self, args):
        self.args = args
        self.dt = 0.1
        self.model = linear_tf_model.LinearTFModel(vars(args), N=args.N, M=args.M, L=args.L, n_steps=args.n_steps)
        self.agent = agent.GazeboAgent(N=args.N, M=args.M, dt=self.dt, model=self.model,
                                       gazebo_model_name=args.model_name)

        rospy.init_node('DAggerMPCAgent')
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)

    def run(self):
        np.random.seed(0)
        joy_msg = Joy()
        joy_msg.axes = [0, 0]

        goal = np.array([[0], [0], [0], [1], [0], [2]])

        # save the models as we retrain
        log_path = experiments_util.experiment_name("dagger_mpc")

        # load our initial model
        self.model.load()

        # used for most of our planning algorithms
        og = self.model.reduce(goal)
        done = False

        try:
            planning_iteration = 0
            new_data = []
            while not done:
                s = self.agent.get_state()
                o = self.model.reduce(s)
                actions, cs, o_s, sbacks = self.agent.greedy_plan(o, goal, T=5)
                planning_iteration += 1

                for i, u in enumerate(actions):
                    joy_msg.axes = [-u[1, 0], -u[0, 0]]

                    # Take action and wait for dt
                    self.unpause(EmptyRequest())
                    self.joy_pub.publish(joy_msg)
                    sleep(self.dt)
                    self.pause(EmptyRequest())

                    s = self.agent.get_state()
                    true_cost = self.agent.state_cost(s, goal)

                    new_data.append(np.concatenate((s.flatten(), u.flatten())))

                    if self.args.pause:
                        input()

                    print("{}, {:0.3f}".format(planning_iteration, true_cost))
                    if true_cost < 0.1:
                        print("Success!")
                        done = True
                        break

                if done:
                    break

                if planning_iteration % 10 == 0:
                    # aggregate data
                    np_x = np.array(new_data)
                    self.model.train(np_x, goal, 1000, log_path)
                    # new_data = []  # clear the array

        except KeyboardInterrupt:
            pass
        finally:
            joy_msg.axes = [0, 0]
            self.joy_pub.publish(joy_msg)


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="load this saved model file")
    parser.add_argument("--model-name", '-m', default="myfirst")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--plot-plan", action='store_true')
    parser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("--n-steps", "-s", type=int, default=1)

    args = parser.parse_args()

    agent = MPCAgent(args)
    agent.run()
