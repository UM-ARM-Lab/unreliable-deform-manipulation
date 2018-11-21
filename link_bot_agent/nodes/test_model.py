#!/usr/bin/env python
from time import sleep
import argparse
import numpy as np
import matplotlib.pyplot as plt

import rospy
from link_bot_notebooks import notebook_finder
from link_bot_notebooks import toy_problem_optimization_common as tpo
from link_bot_notebooks.linear_tf_model import LinearTFModel

from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest, GetLinkState, GetLinkStateRequest
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
        self.apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    def plan_one_step(self, o, goal):
        potential_actions = np.vstack((np.array([[[0], [0]]]), 10 * np.random.randint(-10, 10, size=(200, 2, 1))))
        min_cost_action = None
        next_o = None
        min_cost = 1e9
        colors = []
        xs = []
        ys = []
        for a in potential_actions:
            o_ = self.model.predict_from_o(o, a, self.dt)
            c = self.model.cost_of_o(o_, goal)[0, 0]
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

    def plan(self, o, goal, T=300):
        actions = np.zeros((T, 2, 1))
        os = np.zeros((T, 2))
        sbacks = np.zeros((T, 6))
        for i in range(T):
            s_back = np.linalg.lstsq(self.model.A, o, rcond=None)[0]
            sbacks[i] = np.squeeze(s_back)
            os[i] = np.squeeze(o)
            a, c, next_o = self.plan_one_step(o, goal)
            actions[i] = a
            o = next_o

        plt.figure()
        plt.plot(actions[:, 0, 0], label="x force")
        plt.plot(actions[:, 1, 0], label="y force")
        plt.xlabel("time steps")
        plt.ylabel("force (N)")
        plt.legend()

        plt.figure()
        ax = plt.gca()
        ax.plot(os[:, 0], os[:, 1], label="$o_1$, $o_2$")
        ax.plot(sbacks[:, 0], sbacks[:, 1], label="$s_1$, $s_2$ recovered from $o$")
        S = 20
        q = ax.quiver(sbacks[::S, 0], sbacks[::S, 1], actions[::S, 0, 0], actions[::S, 1, 0], scale=5000, width=0.001)
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
        return np.linalg.norm(s - goal)

    def run(self):
        np.random.seed(111)
        wrench_req = ApplyBodyWrenchRequest()
        wrench_req.body_name = self.model_name + "::head"
        wrench_req.reference_frame = "world"
        wrench_req.duration.secs = 0
        wrench_req.duration.nsecs = self.dt * 1e9

        goal = np.zeros(6)

        # load our initial model
        self.model.load()

        data = []

        prev_s = None
        prev_true_cost = None

        s = self.get_state()
        o = self.model.reduce(s)
        actions = self.plan(o, goal)

        for i, action in enumerate(actions):
            wrench_req.wrench.force.x = action[0, 0]
            wrench_req.wrench.force.y = action[1, 0]

            # Take action and wait for dt
            self.unpause(EmptyRequest())
            self.apply_wrench(wrench_req)
            sleep(self.dt)
            self.pause(EmptyRequest())

            # aggregate data
            s = self.get_state()
            true_cost = self.state_cost(s, goal)
            o_cost = self.model.cost_of_s(s, goal)[0, 0]
            if i > 0:
                datum = np.concatenate(
                    (prev_s.flatten(), action.flatten(), s.flatten(), prev_true_cost.flatten(), true_cost.flatten()))
                if self.args.verbose:
                    print(datum)

                data.append(datum)

            s_back = np.linalg.lstsq(self.model.A, self.model.reduce(s), rcond=None)[0]
            print(s.T, s_back.T, action.T, true_cost, o_cost)
            if self.args.pause:
                input()

            if true_cost < 0.01:
                print("Success!")
                break

            prev_true_cost = true_cost
            prev_s = s

        # Retrain
        # self.pause(EmptyRequest())
        # tpo.train(data, self.model, goal, self.dt, tpo.one_step_cost_prediction_objective)
        # self.unpause(EmptyRequest())

        # Stop the force application
        wrench_req.wrench.force.x = 0
        wrench_req.wrench.force.y = 0
        self.apply_wrench(wrench_req)

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
