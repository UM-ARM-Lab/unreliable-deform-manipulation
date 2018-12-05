#!/usr/bin/env python
from __future__ import print_function

import os
from colorama import Fore
import numpy as np
import tensorflow as tf

from link_bot_notebooks import base_model


def subsequences(x, n_steps):
    s = x.dtype.itemsize
    shape = (n_steps + 1, x.shape[1], x.shape[0] - n_steps)
    strides = (s * x.shape[1], s, s * x.shape[1])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


class LinearTFModel(base_model.BaseModel):

    def __init__(self, args, N, M, L, n_steps, seed=0):
        base_model.BaseModel.__init__(self, N, M, L)

        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        self.args = args
        self.N = N
        self.M = M
        self.L = L
        self.n_steps = n_steps
        self.beta = 1e-8

        self.s = tf.placeholder(tf.float32, shape=(N, None), name="s")
        self.s_ = tf.placeholder(tf.float32, shape=(N, None), name="s_")
        self.u = tf.placeholder(tf.float32, shape=(n_steps, L, None), name="u")
        self.g = tf.placeholder(tf.float32, shape=(N, 1), name="g")
        self.c = tf.placeholder(tf.float32, shape=(None), name="c")
        self.c_ = tf.placeholder(tf.float32, shape=(None), name="c_")

        self.A = tf.Variable(np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]), name="A", dtype=tf.float32)
        self.B = tf.Variable(np.zeros((2, 2)), name="B", dtype=tf.float32)
        self.C = tf.Variable(np.zeros((2, 2)), name="C", dtype=tf.float32)
        self.D = tf.Variable(np.eye(2), name="D", dtype=tf.float32)

        self.hat_o = tf.matmul(self.A, self.s, name='reduce')
        self.og = tf.matmul(self.A, self.g, name='reduce_goal')
        self.o_ = tf.matmul(self.A, self.s_, name='reduce_')

        self.hat_o_ = self.hat_o
        for i in range(self.n_steps):
            self.hat_o_ = self.hat_o_ + tf.matmul(self.B, self.hat_o_, name='dynamics_step_{}'.format(i)) + \
                          tf.matmul(self.C, self.u[i], name='controls_step_{}'.format(i))

        self.d_to_goal = self.og - self.hat_o
        self.d_to_goal_ = self.og - self.hat_o_
        self.hat_c = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal), self.D), self.d_to_goal))
        self.hat_c_ = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal_), self.D), self.d_to_goal_))

        with tf.name_scope("train"):
            self.cost_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            self.state_prediction_loss = tf.reduce_mean(tf.norm(self.o_ - self.hat_o_, axis=0))
            self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c_, predictions=self.hat_c_)
            flat_weights = tf.concat((tf.reshape(self.A, [-1]), tf.reshape(self.B, [-1]),
                                      tf.reshape(self.C, [-1]), tf.reshape(self.D, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(flat_weights) * self.beta
            self.loss = self.cost_loss + self.state_prediction_loss + self.cost_prediction_loss + self.regularization
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            starter_learning_rate = 0.1
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 5000, 0.8,
                                                            staircase=True)
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)

            trainable_vars = tf.trainable_variables()
            grads = zip(tf.gradients(self.loss, trainable_vars), trainable_vars)
            for grad, var in grads:
                name = var.name.replace(":", "_")
                tf.summary.histogram(name + "/gradient", grad)

            tf.summary.scalar("learning_rate", self.learning_rate)
            tf.summary.scalar("cost_loss", self.cost_loss)
            tf.summary.scalar("state_prediction_loss", self.state_prediction_loss)
            tf.summary.scalar("cost_prediction_loss", self.cost_prediction_loss)
            tf.summary.scalar("regularization_loss", self.regularization)
            tf.summary.scalar("loss", self.loss)

            # Set up logging/saving
            self.summaries = tf.summary.merge_all()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.015)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.saver = tf.train.Saver()

    def train(self, train_x, goal, epochs, log_path):
        """
        x train is an array, each row of which looks like:
            [s_t, u_t]
        """
        interrupted = False

        if self.args['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)
            writer = tf.summary.FileWriter(full_log_path)
            writer.add_graph(self.sess.graph)

        try:
            # all sub-trajectories of length n_steps
            seqs = subsequences(train_x, self.n_steps)

            if train_x.shape[0] < self.n_steps:
                raise Exception("Need more time steps of data!")

            s = seqs[0, :self.N, :]
            s_ = seqs[-1, :self.N, :]
            u = seqs[:-1, self.N:, :]
            c = np.sum((seqs[0, [0, 1], :] - goal[[0, 1]]) ** 2, axis=0)
            c_ = np.sum((seqs[-1, [0, 1], :] - goal[[0, 1]]) ** 2, axis=0)
            feed_dict = {self.s: s,
                         self.s_: s_,
                         self.u: u,
                         self.g: goal,
                         self.c: c,
                         self.c_: c_}
            ops = [self.global_step, self.summaries, self.loss, self.opt]
            for i in range(epochs):
                step, summary, loss, _ = self.sess.run(ops, feed_dict=feed_dict)

                if 'print_period' in self.args and step % self.args['print_period'] == 0:
                    print(step, loss)

                if self.args['log'] is not None:
                    writer.add_summary(summary, step)
        except KeyboardInterrupt:
            print("stop!!!")
            interrupted = True
            pass
        finally:
            ops = [self.A, self.B, self.C, self.D]
            A, B, C, D = self.sess.run(ops, feed_dict={})
            if self.args['verbose']:
                print("Loss: {}".format(loss))
                print("A:\n{}".format(A))
                print("B:\n{}".format(B))
                print("C:\n{}".format(C))
                print("D:\n{}".format(D))

            if self.args['log'] is not None:
                self.save(full_log_path)

        return interrupted

    def setup(self):
        if self.args['checkpoint']:
            self.load()
        else:
            self.init()

    def init(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def reduce(self, s):
        feed_dict = {self.s: s}
        ops = [self.hat_o]
        hat_o = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o

    def predict(self, o, u):
        feed_dict = {self.hat_o: o, self.u: u}
        ops = [self.hat_o_]
        hat_o_ = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_

    def predict_from_o(self, o, u, dt=None):
        return self.predict(o, u)

    def cost_of_s(self, s, g):
        return self.cost(self.reduce(s), g)

    def cost(self, o, g):
        feed_dict = {self.hat_o: o, self.g: g}
        ops = [self.hat_c]
        hat_c = self.sess.run(ops, feed_dict=feed_dict)[0]
        hat_c = np.expand_dims(hat_c, axis=0)
        return hat_c

    def act(self, o, g, max_v=1):
        """ return the action which gives the lowest cost for the predicted next state """
        feed_dict = {self.hat_o: o, self.g: g}
        ops = [self.B, self.C, self.og]
        B, C, og = self.sess.run(ops, feed_dict=feed_dict)
        u = np.linalg.lstsq(C, (og - o - np.dot(B, o)), rcond=None)[0]
        u = u.reshape(1, 2, -1)

        if np.linalg.norm(u) < max_v:
            min_cost_u = u
        else:
            min_cost = 1e9
            min_cost_u = None
            for angle in np.linspace(-np.pi, np.pi, 100):
                u = np.array([[[np.cos(angle)], [np.sin(angle)]]])
                cost = self.predict_cost(o, u, g)[0, 0]
                if cost < min_cost:
                    min_cost = cost
                    min_cost_u = u

        feed_dict = {self.hat_o: o, self.g: g, self.u: min_cost_u}
        ops = [self.hat_o_, self.hat_c_]
        hat_o_, hat_c_ = self.sess.run(ops, feed_dict=feed_dict)
        return min_cost_u, hat_c_, hat_o_

    def save(self, log_path):
        global_step = self.sess.run(self.global_step)
        print(Fore.CYAN + "Saving ckpt {} at step {:d}".format(log_path, global_step) + Fore.RESET)
        self.saver.save(self.sess, os.path.join(log_path, "nn.ckpt"), global_step=self.global_step)

    def load(self):
        self.saver.restore(self.sess, self.args['checkpoint'])
        global_step = self.sess.run(self.global_step)
        print(Fore.CYAN + "Restored ckpt {} at step {:d}".format(self.args['checkpoint'], global_step) + Fore.RESET)

    def evaluate(self, eval_x, goal, display=True):
        seqs = subsequences(eval_x, self.n_steps)
        s = seqs[0, :self.N, :]
        s_ = seqs[-1, :self.N, :]
        u = seqs[:-1, self.N:, :]
        c = np.sum((seqs[0, [0, 1], :] - goal[[0, 1]]) ** 2, axis=0)
        c_ = np.sum((seqs[-1, [0, 1], :] - goal[[0, 1]]) ** 2, axis=0)
        feed_dict = {self.s: s,
                     self.s_: s_,
                     self.u: u,
                     self.g: goal,
                     self.c: c,
                     self.c_: c_}
        ops = [self.A, self.B, self.C, self.D, self.o_, self.hat_o_, self.cost_loss, self.state_prediction_loss,
               self.cost_prediction_loss,
               self.regularization, self.loss]
        A, B, C, D, o_, o_hat_, c_loss, sp_loss, cp_loss, reg, loss = self.sess.run(ops, feed_dict=feed_dict)
        if display:
            print("Cost Loss: {}".format(c_loss))
            print("State Prediction Loss: {}".format(sp_loss))
            print("Cost Prediction Loss: {}".format(cp_loss))
            print("Regularization: {}".format(reg))
            print("Overall Loss: {}".format(loss))
            print("A:\n{}".format(A))
            print("B:\n{}".format(B))
            print("C:\n{}".format(C))
            print("D:\n{}".format(D))
        return A, B, C, D, c_loss, sp_loss, cp_loss, reg, loss

    def get_A(self):
        feed_dict = {}
        ops = [self.A]
        A = self.sess.run(ops, feed_dict=feed_dict)[0]
        return A
