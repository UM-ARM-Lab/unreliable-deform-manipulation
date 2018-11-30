#!/usr/bin/env python
from time import sleep
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import Joy
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest
from std_srvs.srv import Empty, EmptyRequest
import rospy

from link_bot_notebooks.linear_tf_model import LinearTFModel
from link_bot_agent import a_star
from link_bot_agent import gz_world


def h(n1, n2):
    return np.linalg.norm(np.array(n1) - np.array(n2))


class TestModel:

    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.dt = 0.1
        self.model = LinearTFModel({'checkpoint': self.args.checkpoint}, N=args.N, M=args.M, L=args.L)

        rospy.init_node('ScipyLearner')
        self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)

    def sample_action(self, o, goal):
        potential_actions = 0.3 * np.random.randn(250, 2, 1)
        min_cost_action = None
        next_o = None
        min_cost = 1e9
        xs = []
        ys = []
        for a in potential_actions:
            o_ = self.model.predict_from_o(o, a, dt=self.dt)
            c = self.model.cost(o_, goal)[0, 0]
            x = o_[0, 0]
            y = o_[1, 0]
            xs.append(x)
            ys.append(y)
            if c < min_cost:
                min_cost = c
                next_o = o_
                min_cost_action = a

        return min_cost_action, min_cost, next_o

    def greedy_action(self, o, goal):
        MAX_SPEED = 1
        full_u, full_c, next_o = self.model.act(o, goal)
        if np.linalg.norm(full_u) > MAX_SPEED:
            u = full_u / np.linalg.norm(full_u) * MAX_SPEED  # u is in meters per second. Cap to 0.75
        else:
            u = full_u
        c = self.model.predict_cost(o, u, goal)
        next_o = self.model.predict(o, u)
        return u, c, next_o

    def a_star_plan(self, o, og):

        # construct our graph as a list of edges and vertices
        graph = gz_world.GzWorldGraph()
        planner = a_star.AStar(graph, h)
        shortest_path = planner.shortest_path(o, og)

        T = len(shortest_path)
        actions = np.zeros((T, 2, 1))
        os = np.zeros((T, self.args.M))
        cs = np.zeros(T)
        sbacks = np.zeros((T, self.args.N))
        for i, o in enumerate(shortest_path):
            s_back = np.linalg.lstsq(self.model.get_A(), o, rcond=None)[0]
            sbacks[i] = np.squeeze(s_back)
            os[i] = np.squeeze(o)
            cs[i] = c
            actions[i] = u

        return actions, cs, os, sbacks

    def greedy_plan(self, o, goal, T=1):
        actions = np.zeros((T, 2, 1))
        os = np.zeros((T, self.args.M))
        cs = np.zeros(T)
        sbacks = np.zeros((T, self.args.N))

        for i in range(T):
            s_back = np.linalg.lstsq(self.model.get_A(), o, rcond=None)[0]
            sbacks[i] = np.squeeze(s_back)
            os[i] = np.squeeze(o)

            # u, c, next_o = self.sample_action(o, goal)
            u, c, next_o = self.greedy_action(o, goal)

            cs[i] = c
            actions[i] = u
            o = next_o

        return actions, cs, os, sbacks

    def get_state(self):
        o = []
        links = ['link_0', 'link_1', 'head']
        for link in links:
            link_state_req = GetLinkStateRequest()
            link_state_req.link_name = link
            link_state_resp = self.get_link_state(link_state_req)
            link_state = link_state_resp.link_state
            x = link_state.pose.position.x
            y = link_state.pose.position.y
            o.extend([x, y])
        return np.expand_dims(o, axis=1)

    @staticmethod
    def state_cost(s, goal):
        return np.linalg.norm(s[0:2, 0] - goal[0:2, 0])

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
                s = self.get_state()
                o = self.model.reduce(s)
                actions, cs, os, sbacks = self.greedy_plan(o, goal)
                # actions, cs, os, sbacks = self.a_star_plan(o, og)

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
                    # ax.plot(os[:, 0], os[:, 1], label="$o_1$, $o_2$")
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

                    s = self.get_state()
                    true_cost = self.state_cost(s, goal)

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

    args = parser.parse_args()

    agent = TestModel(args)
    agent.run()
