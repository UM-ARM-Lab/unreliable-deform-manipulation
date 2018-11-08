#!/usr/bin/env python

import tensorflow as tf
import os
from datetime import datetime
from time import sleep
import argparse
import numpy as np

import rospy
from link_bot_notebooks import notebook_finder
from link_bot_notebooks import model_common
from link_bot_notebooks import toy_problem_optimization_common as tpo

from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest, GetLinkState, GetLinkStateRequest
from std_srvs.srv import Empty, EmptyRequest


class NNModel(model_common.BaseModel):

    def __init__(self, N, M, L, n_hidden):
        super(NNModel).__init__(self, N, M, L)

    def reduce(self, s):
        pass

    def predict(self, o, u):
        pass

    def cost(self, o, goal):
        pass

    def train(self, x_train, y_train, epochs=5):
        """
        x train is an array, each row of which looks like:
            [s_t, u_t, s_{t+1}, goal]
        y train is an array, each row of which looks like:
            [c_t, c_{t+1}]
         """
        pass


class NNLearner:

    def __init__(self, args):
        self.args = args
        self.wrapper_name = args.model_name
        self.dt = 0.1
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

        goal = np.array([[4], [0], [5], [0], [6], [0]])

        # set up our model
        summaries = tf.summary.merge_all()
        sess = tf.Session()
        saver = tf.train.Saver()
        stamp = "{:%B_%d_%H:%M:%S}".format(datetime.now())
        log_dir = os.path.join("log_data", stamp)
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(sess.graph)

        # load our initial training data and train to it
        if self.args.load:
            saver.restore(sess, self.args.load)
        else:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            self.pause(EmptyRequest())

            log_data = np.loadtxt(self.args.initial_data)
            n_training_samples = log_data.shape[0]
            initial_x = np.ndarray(n_training_samples - 1, 6 + 6 + 2 + 6)
            initial_y = np.ndarray(n_training_samples - 1, 2)
            for i in range(n_training_samples - 1):
                s = np.expand_dims(log_data[i][0:6], axis=1)
                s_ = np.expand_dims(log_data[i + 1][0:6], axis=1)
                u = np.expand_dims(log_data[i][8:10], axis=1)
                c = (log_data[i][0] - g[0]) ** 2 + (log_data[i][1] - g[1]) ** 2
                c_ = (log_data[i + 1][0] - g[0]) ** 2 + (log_data[i + 1][1] - g[1]) ** 2
                initial_x[i][0:6] = s
                initial_x[i][6:12] = s_
                initial_x[i][12:14] = u
                initial_x[i][14:20] = goal
                initial_y[i][0] = c
                initial_y[i][1] = c_

            batch_size = 32
            for i in range(self.args.epochs):
                start = np.random.randint(0, n_training_samples - batch_size)
                batch_x = initial_x[start, start + batch_size][0, 1, 2]
                batch_y = initial_y[start, start + batch_size]
                feed_dict = {self.nn.x: batch_x, self.nn.y: batch_y}
                ops = [self.nn.global_step, summaries, self.nn.loss]
                step, summary, loss = sess.run(ops, feed_dict=feed_dict)
                writer.add_summary(summary, step)

            saver.save(sess, os.path.join(log_dir, "nn.ckpt"))

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
