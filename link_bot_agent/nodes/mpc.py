#!/usr/bin/env python
from time import sleep
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import Joy
from std_srvs.srv import Empty, EmptyRequest
import rospy

from link_bot_notebooks.linear_tf_model import LinearTFModel
import agent


def h(n1, n2):
    return np.linalg.norm(np.array(n1) - np.array(n2))


class MPCAgent:

    def __init__(self, args):
        self.args = args
        self.dt = 0.1
        self.model = LinearTFModel(vars(args), N=args.N, M=args.M, L=args.L, n_steps=args.n_steps)
        self.agent = agent.GazeboAgent(N=args.N, M=args.M, dt=self.dt, model=self.model,
                                       gazebo_model_name=args.model_name)

        rospy.init_node('MPCAgent')
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)

    def run(self):
        np.random.seed(0)
        joy_msg = Joy()
        joy_msg.axes = [0, 0]

        goal = np.array([[0], [0], [0], [1], [0], [2]])

        # load our initial model
        self.model.load()

        # used for most of our planning algorithms
        og = self.model.reduce(goal)
        done = False

        try:
            while not done:
                s = self.agent.get_state()
                o = self.model.reduce(s)
                actions, cs, os, sbacks = self.agent.greedy_plan(o, goal)
                # actions, cs, os, sbacks = self.agent.a_star_plan(o, og)

                if args.plot_plan:
                    plt.figure()
                    plt.plot(cs)
                    plt.xlabel("time steps")
                    plt.ylabel("predicted cost (squared distance, m^2)")
                    plt.legend()

                    plt.figure()
                    plt.plot(actions[:, 0, 0], label="x velocity")
                    plt.plot(actions[:, 1, 0], label="y velocity")
                    plt.xlabel("time steps")
                    plt.ylabel("velocity (m/s)")
                    plt.legend()

                    plt.figure()
                    ax = plt.gca()
                    ax.plot(sbacks[:, 0], sbacks[:, 1], label="$s_1$, $s_2$ recovered from $o$")
                    S = 10
                    q = ax.quiver(sbacks[::S, 0], sbacks[::S, 1], actions[::S, 0, 0], actions[::S, 1, 0], scale=100,
                                  width=0.002)
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Y (m)")
                    ax.set_aspect('equal')
                    plt.legend()
                    plt.show()
                    return

                for i, action in enumerate(actions):
                    joy_msg.axes = [-action[1, 0], -action[0, 0]]

                    # Take action and wait for dt
                    self.unpause(EmptyRequest())
                    self.joy_pub.publish(joy_msg)
                    sleep(self.dt)
                    self.pause(EmptyRequest())

                    s = self.agent.get_state()
                    true_cost = self.agent.state_cost(s, goal)

                    if self.args.pause:
                        input()

                    print("{:0.3f}".format(true_cost))
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
            self.joy_pub.publish(joy_msg)


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="load this saved model file")
    parser.add_argument("--model-name", '-m', default="myfirst")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--plot-plan", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("--n-steps", "-s", type=int, default=1)

    args = parser.parse_args()

    agent = MPCAgent(args)
    agent.run()
