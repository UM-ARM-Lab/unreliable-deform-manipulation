#!/usr/bin/env python
from time import sleep
import argparse
import numpy as np
import matplotlib.pyplot as plt

import rospy
from link_bot_notebooks import notebook_finder
from link_bot_notebooks import toy_problem_optimization_common as tpo
from link_bot_notebooks.linear_tf_model import LinearTFModel

from sensor_msgs.msg import Joy
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest
from std_srvs.srv import Empty, EmptyRequest


class TestModel:

    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.dt = 0.1
        # self.model = tpo.LinearStateSpaceModelWithQuadraticCost(N=6, M=2, L=2)
        self.model = LinearTFModel({'checkpoint': self.args.checkpoint}, N=6, M=2, L=2)

        rospy.init_node('ScipyLearner')
        self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)

    def plan_one_step(self, o, goal):
        potential_actions = np.vstack((np.array([[[0], [0]]]), 0.1 * np.random.randint(-10, 10, size=(200, 2, 1))))
        min_cost_action = None
        next_o = None
        min_cost = 1e9
        colors = []
        xs = []
        ys = []
        for a in potential_actions:
            o_ = self.model.predict_from_o(o, a, dt=self.dt)
            c = self.model.cost(o_, goal)[0, 0]
            x = o_[0, 0]
            y = o_[1, 0]
            xs.append(x)
            ys.append(y)
            colors.append(c)
            if c < min_cost:
                min_cost = c
                next_o = o_
                min_cost_action = a

        return min_cost_action, min_cost, next_o

    def plan(self, o, goal, T=10, plot=False):
        actions = np.zeros((T, 2, 1))
        os = np.zeros((T, 2))
        cs = np.zeros(T)
        sbacks = np.zeros((T, 6))
        for i in range(T):
            s_back = np.linalg.lstsq(self.model.get_A(), o, rcond=None)[0]
            sbacks[i] = np.squeeze(s_back)
            os[i] = np.squeeze(o)
            # u, c, next_o = self.plan_one_step(o, goal)
            # guess_u, guess_c, guess_next_o = self.plan_one_step(o, goal)
            full_u, full_c, next_o = self.model.act(o, goal)
            if np.linalg.norm(full_u) > 1.00:
                u = full_u / np.linalg.norm(full_u) * 1.00  # u is in meters per second. Cap to 0.75
            else:
                u = full_u
            c = self.model.predict_cost(o, u, goal)
            next_o = self.model.predict(o, u)
            cs[i] = c
            # print(guess_u, guess_c)
            # print(full_u, full_c)
            # print(u, c)
            actions[i] = u
            o = next_o

        if plot:
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
            ax.plot(os[:, 0], os[:, 1], label="$o_1$, $o_2$")
            ax.plot(sbacks[:, 0], sbacks[:, 1], label="$s_1$, $s_2$ recovered from $o$")
            S = 10
            q = ax.quiver(sbacks[::S, 0], sbacks[::S, 1], actions[::S, 0, 0], actions[::S, 1, 0], scale=5000,
                          width=0.001)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            plt.legend()
            plt.show()

        return actions

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
        np.random.seed(111)
        joy_msg = Joy()
        joy_msg.axes = [0, 0]

        goal = np.array([[0], [0], [1], [0], [2], [0]])

        # load our initial model
        self.model.load()

        data = []

        prev_s = None
        prev_true_cost = None
        done = False

        try:
            self.unpause(EmptyRequest())
            while not done:
                s = self.get_state()
                o = self.model.reduce(s)
                actions = self.plan(o, goal)
                for i, action in enumerate(actions):
                    joy_msg.axes = [-action[1, 0], -action[0, 0]]

                    # Take action and wait for dt
                    # self.unpause(EmptyRequest())
                    self.joy_pub.publish(joy_msg)
                    sleep(self.dt)
                    # self.pause(EmptyRequest())

                    # aggregate data
                    s = self.get_state()
                    true_cost = self.state_cost(s, goal)
                    # if i > 0:
                    #     datum = np.concatenate((prev_s.flatten(), action.flatten(), s.flatten(), prev_true_cost.flatten(), true_cost.flatten()))
                    #     if self.args.verbose:
                    #         print(datum)
                    #
                    #     data.append(datum)

                    # s_back = np.linalg.lstsq(self.model.get_A(), o, rcond=None)[0]
                    if self.args.pause:
                        input()

                    print("{:0.3f}".format(true_cost))
                    if true_cost < 0.1:
                        print("Success!")
                        done = True
                        break

                    prev_true_cost = true_cost
                    prev_s = s
                if done:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            joy_msg.axes = [0, 0]
            self.joy_pub.publish(joy_msg)
            if self.args.new_data:
                np.savetxt(self.args.new_data, data)


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="load this saved model file")
    parser.add_argument("--model-name", '-m', default="myfirst")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--new_data", help='filename to store data in')

    args = parser.parse_args()

    agent = TestModel(args)
    agent.run()
