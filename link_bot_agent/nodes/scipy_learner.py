#!/usr/bin/env python
from time import sleep

from link_bot_notebooks import notebook_finder
from link_bot_notebooks import toy_problem_optimization_common as tpo
import argparse
import numpy as np

import rospy
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest, GetLinkState, GetLinkStateRequest


class ScipyLearner:

    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.dt = 0.1
        self.model = tpo.LinearStateSpaceModelWithQuadraticCost(N=4, M=2, L=2)

        rospy.init_node('ScipyLearner')
        self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        self.apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

    def plan(self, o, goal):
        potential_actions = np.random.randn(20, 2, 1)
        min_cost_action = potential_actions[0]
        min_cost = self.model.predict_cost(o, potential_actions[0], self.dt, goal)
        for a in potential_actions:
            c = self.model.predict_cost(o, potential_actions[0], self.dt, goal)
            if c < min_cost:
                min_cost = c
                min_cost_action = a

        return min_cost_action

    def get_state(self):
        o = []
        links = ['link_0', 'link_1']
        for link in links:
            link_state_req = GetLinkStateRequest()
            link_state_req.link_name = link
            link_state_resp = self.get_link_state(link_state_req)
            link_state = link_state_resp.link_state
            x = link_state.pose.position.x
            y = link_state.pose.position.y
            o.extend([x, y])
        return np.expand_dims(o, axis=1)

    def get_action(self):
        link_state_req = GetLinkStateRequest()
        link_state_req.link_name = 'link_1'
        link_state_resp = self.get_link_state(link_state_req)
        link_state = link_state_resp.link_state
        vx = link_state.twist.linear.x
        vy = link_state.twist.linear.y
        return np.array([[vx], [vy]])

    @staticmethod
    def state_cost(s, goal):
        return np.linalg.norm(s - goal)

    def run(self):
        wrench_req = ApplyBodyWrenchRequest()
        wrench_req.body_name = self.model_name + "::link_1"
        wrench_req.reference_frame = "world"
        wrench_req.duration.secs = -1
        wrench_req.duration.nsecs = -1

        data = []
        goal = np.array([[5], [0], [6], [0]])

        # get the current state of the world and reduce it
        s = self.get_state()
        o = self.model.reduce(s)
        cost = self.model.cost_of_o(o, goal)
        init = False

        for i in range(100):
            if init:
                tpo.train(data, self.model, goal, self.dt, tpo.one_step_cost_prediction_objective)
            init = True

            action = self.plan(o, goal)
            print(action)

            wrench_req.wrench.force.x = action[0, 0]
            wrench_req.wrench.force.y = action[1, 0]

            # Take action and wait for dt
            self.apply_wrench(wrench_req)
            sleep(self.dt)

            # aggregate data
            prev_s = s
            prev_cost = cost
            s = self.get_state()
            action = self.get_action()
            cost = self.state_cost(s, goal)
            datum = [prev_s, action, s, prev_cost, cost]

            if self.args.verbose:
                print(datum)
            data.append(datum)

        # Stop the force application
        wrench_req.wrench.force.x = 0
        wrench_req.wrench.force.y = 0
        self.apply_wrench(wrench_req)

        np.savetxt(self.args.outfile, data)


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", '-m', default="link_bot")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("outfile", help='filename to store data in')

    args = parser.parse_args()

    agent = ScipyLearner(args)
    agent.run()
