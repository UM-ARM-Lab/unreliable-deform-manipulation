#!/usr/bin/env python

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from link_bot_notebooks import base_model


def load_and_construct_training_data(filename, N, M, L, goal):
    log_data = np.loadtxt(filename)
    n_training_samples = log_data.shape[0]
    train_x = np.ndarray((n_training_samples - 1, 3 * N + L))
    train_y = np.ndarray((n_training_samples - 1, 2))
    for i in range(n_training_samples - 1):
        s = log_data[i][0:6]
        s_ = log_data[i + 1][0:6]
        u = log_data[i][8:10]
        c = (log_data[i][0] - goal[0]) ** 2 + (log_data[i][1] - goal[1]) ** 2
        c_ = (log_data[i + 1][0] - goal[0]) ** 2 + (log_data[i + 1][1] - goal[1]) ** 2
        train_x[i] = np.concatenate((s, s_, goal, u))
        train_y[i][0] = c
        train_y[i][1] = c_

    return n_training_samples, train_x, train_y


class NNModel(base_model.BaseModel):

    def __init__(self, args, N, M, L, dims):
        base_model.BaseModel.__init__(self, N, M, L)

        self.args = args
        self.N = N
        self.M = M
        self.L = L

        # set up our model

        self.s = tf.placeholder(tf.float32, shape=(None, N), name="s")
        self.s_ = tf.placeholder(tf.float32, shape=(None, N), name="s_")
        self.u = tf.placeholder(tf.float32, shape=(None, L), name="u")
        self.g = tf.placeholder(tf.float32, shape=(None, N), name="g")
        self.c = tf.placeholder(tf.float32, shape=(None), name="c")
        self.c_ = tf.placeholder(tf.float32, shape=(None), name="c_")

        self.F1_w = tf.Variable(tf.truncated_normal([N, dims['F1_n']]), dtype=tf.float32, name='F1_n')
        self.F2_w = tf.Variable(tf.truncated_normal([dims['F1_n'], dims['F2_n']]), dtype=tf.float32, name='F2_n')
        self.F3_w = tf.Variable(tf.truncated_normal([dims['F2_n'], M]), dtype=tf.float32, name='F3_n')

        self.F1_b = tf.Variable(tf.truncated_normal([dims['F1_n']]), dtype=tf.float32, name='F1_n')
        self.F2_b = tf.Variable(tf.truncated_normal([dims['F2_n']]), dtype=tf.float32, name='F2_n')
        self.F3_b = tf.Variable(tf.truncated_normal([M]), dtype=tf.float32, name='F3_n')

        self.T1_w = tf.Variable(tf.truncated_normal([M + L, dims['T1_n']]), dtype=tf.float32, name='T1_n')
        self.T2_w = tf.Variable(tf.truncated_normal([dims['T1_n'], dims['T2_n']]), dtype=tf.float32, name='T2_n')
        self.T3_w = tf.Variable(tf.truncated_normal([dims['T2_n'], M]), dtype=tf.float32, name='T3_n')

        self.T1_b = tf.Variable(tf.truncated_normal([dims['T1_n']]), dtype=tf.float32, name='T1_n')
        self.T2_b = tf.Variable(tf.truncated_normal([dims['T2_n']]), dtype=tf.float32, name='T2_n')
        self.T3_b = tf.Variable(tf.truncated_normal([M]), dtype=tf.float32, name='T3_n')

        self.C_w = tf.Variable(tf.truncated_normal([M, M]), dtype=tf.float32, name='C_n')

        def make_F(F1_input, name):
            with tf.name_scope("Func_F" + name):
                F2_input = tf.nn.relu(tf.matmul(F1_input, self.F1_w) + self.F1_b)
                F3_input = tf.nn.relu(tf.matmul(F2_input, self.F2_w) + self.F2_b)
                F_output = tf.nn.softmax(tf.matmul(F3_input, self.F3_w) + self.F3_b)
                return F_output

        def make_T(T1_input, U_input, name):
            with tf.name_scope("Func_T" + name):
                T2_input = tf.nn.relu(tf.matmul(tf.concat([T1_input, U_input], 1), self.T1_w) + self.T1_b)
                T3_input = tf.nn.relu(tf.matmul(T2_input, self.T2_w) + self.T2_b)
                T_output = tf.nn.softmax(tf.matmul(T3_input, self.T3_w) + self.T3_b)
                return T_output

        def make_C(C_input, G_input, name):
            with tf.name_scope("Func_C" + name):
                C_w_normalized = tf.clip_by_value(self.C_w, -1, 1)
                C_output = tf.matmul(tf.matmul((G_input - C_input), C_w_normalized), tf.transpose(C_input - G_input))
                return C_output

        self.hat_o = make_F(self.s, name="")
        self.og = make_F(self.g, name="g")
        self.o_ = make_F(self.s_, name="_")

        self.hat_o_ = make_T(self.hat_o, self.u, name="s")

        self.hat_c = make_C(self.hat_o, self.og, name="s")
        self.hat_c_ = make_C(self.hat_o_, self.og, name="")

        with tf.name_scope("train"):
            self.cost_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            self.state_prediction_loss = tf.losses.mean_squared_error(labels=self.o_, predictions=self.hat_o_)
            self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c_, predictions=self.hat_c_)
            self.loss = self.cost_loss + self.state_prediction_loss + self.cost_prediction_loss
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss, global_step=self.global_step)
            trainable_vars = tf.trainable_variables()
            grads = zip(tf.gradients(self.loss, trainable_vars), trainable_vars)
            for grad, var in grads:
                tf.summary.histogram(var.name + "/gradient", grad)

        tf.summary.scalar("loss", self.loss)

        # Set up logging/saving
        self.summaries = tf.summary.merge_all()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        stamp = "{:%B_%d_%H:%M:%S}".format(datetime.now())
        self.log_dir = None
        if self.args.log:
            self.log_dir = os.path.join("log_data", stamp)
            self.writer = tf.summary.FileWriter(self.log_dir)
            self.writer.add_graph(self.sess.graph)

    def reduce(self, s):
        feed_dict = {self.s: s}
        ops = [self.o]
        o = self.sess.run(ops, feed_dict=feed_dict)
        return o

    def predict(self, o, u):
        feed_dict = {self.o: o, self.u: u}
        ops = [self.o_]
        o_ = self.sess.run(ops, feed_dict=feed_dict)
        return o_

    def cost(self, o, goal):
        feed_dict = {self.o: o, self.g: goal}
        ops = [self.c]
        c = self.sess.run(ops, feed_dict=feed_dict)
        return c

    def train(self, x_train, y_train, epochs, checkpoint=None, seed=0):
        """
        x train is an array, each row of which looks like:
            [s_t, u_t, s_{t+1}, goal]
        y train is an array, each row of which looks like:
            [c_t, c_{t+1}]
        """
        # load our train training data and train to it
        np.random.seed(seed)
        if checkpoint:
            self.load(checkpoint)
        else:
            self.init()

        n_training_samples = x_train.shape[0]
        batch_size = 32
        for i in range(epochs):
            start = np.random.randint(0, n_training_samples - batch_size)
            batch_s = x_train[start: start + batch_size, 0:self.N]
            batch_s_ = x_train[start: start + batch_size, self.N: 2 * self.N]
            batch_g = x_train[start: start + batch_size, 2 * self.N: 3 * self.N]
            batch_u = x_train[start: start + batch_size, 3 * self.N: 3 * self.N + self.L]
            batch_c = y_train[start: start + batch_size, 0]
            batch_c_ = y_train[start: start + batch_size, 1]
            feed_dict = {self.s: batch_s,
                         self.s_: batch_s_,
                         self.u: batch_u,
                         self.g: batch_g,
                         self.c: batch_c,
                         self.c_: batch_c_}
            ops = [self.global_step, self.summaries, self.loss, self.opt]
            step, summary, loss, _ = self.sess.run(ops, feed_dict=feed_dict)

            if step % 10 == 0:
                print(step, loss)

            if self.args.log:
                self.writer.add_summary(summary, step)

        if self.args.log:
            self.saver.save(self.sess, os.path.join(self.log_dir, "nn.ckpt"))

    def init(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def save(self):
        self.saver.save(self.sess, os.path.join(self.log_dir, "nn.ckpt"))

    def load(self, checkpoint):
        self.saver.restore(self.sess, checkpoint)
