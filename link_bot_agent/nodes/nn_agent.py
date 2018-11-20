#!/usr/bin/env python

from time import sleep
import argparse
import numpy as np

import rospy
from link_bot_notebooks import notebook_finder
from link_bot_notebooks import model_common
from link_bot_notebooks import toy_problem_optimization_common as tpo

from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest, GetLinkState, GetLinkStateRequest
from std_srvs.srv import Empty, EmptyRequest


class NNLearner:

    def __init__(self, args):
        self.args = args
        self.wrapper_name = args.model_name
        self.dt = 0.1
        goal = np.array([[4], [0], [5], [0], [6], [0]])
        self.nn = NNModel()
        self.wrapper = model_common.ModelWrapper(self.nn)

        rospy.init_node('NNLearner')
        self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        self.apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    def plan(self, o, goal):
        potential_actions = np.random.randint(-100, 100, (100, 2, 1))
        min_cost_action = potential_actions[0]
        min_cost = self.wrapper.predict_cost(o, potential_actions[0], self.dt, goal)
        for a in potential_actions:
            c = self.wrapper.predict_cost(o, potential_actions[0], self.dt, goal)
            if c < min_cost:
                min_cost = c
                min_cost_action = a

        return min_cost_action

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
        np.random.seed(123)
        wrench_req = ApplyBodyWrenchRequest()
        wrench_req.body_name = self.wrapper_name + "::head"
        wrench_req.reference_frame = "world"
        wrench_req.duration.secs = -1
        wrench_req.duration.nsecs = -1


        # load our initial training data and train to it
        self.pause(EmptyRequest())
        self.unpause(EmptyRequest())

        data = []

        # get the current state of the world and reduce it
        s = self.get_state()
        o = self.wrapper.reduce(s)
        cost = self.wrapper.cost_of_o(o, goal)

        for i in range(1000):
            action = self.plan(o, goal)

            wrench_req.wrench.force.x = action[0, 0]
            wrench_req.wrench.force.y = action[1, 0]

            # Take action and wait for dt
            self.apply_wrench(wrench_req)
            sleep(self.dt)

            # aggregate data
            prev_s = s
            prev_cost = cost
            s = self.get_state()
            cost = self.state_cost(s, goal)
            datum = np.concatenate((prev_s.flatten(),
                                    action.flatten(),
                                    s.flatten(),
                                    prev_cost.flatten(),
                                    cost.flatten()))

            if self.args.verbose:
                print(datum)
            data.append(datum)

            if cost < 1e-2:
                print("Success!")
                break

            # Retrain
            # self.pause(EmptyRequest())
            # tpo.train(data, self.wrapper, goal, self.dt, tpo.one_step_cost_prediction_objective)
            # self.unpause(EmptyRequest())

        # Stop the force application
        wrench_req.wrench.force.x = 0
        wrench_req.wrench.force.y = 0
        self.apply_wrench(wrench_req)

        np.savetxt(self.args.outfile, data)


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", '-m', default="myfirst")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("initial_data", help="initial training data file")
    parser.add_argument("model_save_file", help="initial training data file")
    parser.add_argument("outfile", help='filename to store data in')
    parser.add_argument("--load", help="load this saved model file")
    parser.add_argument("--epochs", '-e', type=int, default=10, help="number of epochs to train for")

    args = parser.parse_args()

    agent = NNLearner(args)
    agent.run()
