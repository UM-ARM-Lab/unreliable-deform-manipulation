#!/usr/bin/env python

import tensorflow as tf
import os
from datetime import datetime
import argparse
import numpy as np

from link_bot_notebooks import base_model


def load_and_construct_training_data(filename, N, M, L, goal):
    log_data = np.loadtxt(filename)
    n_training_samples = log_data.shape[0]
    train_x = np.ndarray(n_training_samples - 1, 3*N + L)
    train_y = np.ndarray(n_training_samples - 1, 2)
    for i in range(n_training_samples - 1):
        s = np.expand_dims(log_data[i][0:6], axis=1)
        s_ = np.expand_dims(log_data[i + 1][0:6], axis=1)
        u = np.expand_dims(log_data[i][8:10], axis=1)
        c = (log_data[i][0] - g[0]) ** 2 + (log_data[i][1] - g[1]) ** 2
        c_ = (log_data[i + 1][0] - g[0]) ** 2 + (log_data[i + 1][1] - g[1]) ** 2
        train_x[i][0:N] = s
        train_x[i][N:2*N] = s_
        train_x[i][2*N:2*N+L] = u
        train_x[i][2*N+L:3*N+L] = goal
        train_y[i][0] = c
        train_y[i][1] = c_

    return n_training_samples, train_x, train_y


class NNModel(base_model.BaseModel):

    def __init__(self, N, M, L, dimensions):
        super(NNModel, self).__init__(N, M, L)

        # set up our model
        self.summaries = tf.summary.merge_all()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        stamp = "{:%B_%d_%H:%M:%S}".format(datetime.now())
        self.log_dir = os.path.join("log_data", stamp)
        self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer.add_graph(self.sess.graph)

        self.s = tf.placeholder(tf.float32, shape=(None, N), name="s")
        self.s_ = tf.placeholder(tf.float32, shape=(None, N), name="s_")
        self.u = tf.placeholder(tf.float32, shape=(None, L), name="u")
        self.g = tf.placeholder(tf.float32, shape=(None, N), name="g")
        self.c = tf.placeholder(tf.float32, shape=(None, 1), name="c")
        self.c_ = tf.placeholder(tf.float32, shape=(None, 1), name="c_")

        self.F1_w = tf.Variable(tf.truncated_normal(dimensions['F1_w']), dtype=tf.float32, name='F1_w')
        self.F2_w = tf.Variable(tf.truncated_normal(dimensions['F2_w']), dtype=tf.float32, name='F2_w')
        self.F3_w = tf.Variable(tf.truncated_normal(dimensions['F3_w']), dtype=tf.float32, name='F3_w')
        self.F1_b = tf.Variable(tf.truncated_normal(dimensions['F1_b']), dtype=tf.float32, name='F1_b')
        self.F2_b = tf.Variable(tf.truncated_normal(dimensions['F2_b']), dtype=tf.float32, name='F2_b')
        self.F3_b = tf.Variable(tf.truncated_normal(dimensions['F3_b']), dtype=tf.float32, name='F3_b')

        self.T1_w = tf.Variable(tf.truncated_normal(dimensions['T1_w']), dtype=tf.float32, name='T1_w')
        self.T2_w = tf.Variable(tf.truncated_normal(dimensions['T2_w']), dtype=tf.float32, name='T2_w')
        self.T3_w = tf.Variable(tf.truncated_normal(dimensions['T3_w']), dtype=tf.float32, name='T3_w')
        self.T1_b = tf.Variable(tf.truncated_normal(dimensions['T1_b']), dtype=tf.float32, name='T1_b')
        self.T2_b = tf.Variable(tf.truncated_normal(dimensions['T2_b']), dtype=tf.float32, name='T2_b')
        self.T3_b = tf.Variable(tf.truncated_normal(dimensions['T3_b']), dtype=tf.float32, name='T3_b')

        self.C_w = tf.Variable(tf.truncated_normal(dimensions['C_w']), dtype=tf.float32, name='C_w')

        def make_F(F1_input, name):
            with tf.name_scope("F_"+name):
                F2_input = tf.nn.relu(tf.matmul(self.F1_w, F1_input) + self.F1_b)
                F3_input = tf.nn.relu(tf.matmul(self.F2_w, F2_input) + self.F2_b)
                F_output = tf.nn.softmax(tf.matmul(self.F3_w, F3_input) + self.F3_b)
                return F_output

        def make_T(T1_input, U_input, name):
            with tf.name_scope("T_"+name):
                T2_input = tf.nn.relu(tf.matmul(self.T1_w, tf.concat((T1_input, U_input))) + self.T1_b)
                T3_input = tf.nn.relu(tf.matmul(self.T2_w, T2_input) + self.T2_b)
                T_output = tf.nn.softmax(tf.matmul(self.T3_w, T3_input) + self.T3_b)
                return T_output

        def make_C(C_input, G_input, name):
            with tf.name_scope("C_"+name):
                C_output = tf.matmul(tf.matmul((G_input - C_input), self.C_w),  tf.transpose(C_input - G_input))
                return C_output

        self.o = make_F(self.s, name="s")
        self.o_g = make_F(self.g, name="g")
        self.o_ = make_F(self.s_, name="s_")

        self.hat_o_ = make_T(self.o, self.u, name="s")

        self.hat_c_o_ = make_C(self.o, self.o_g, name="s_")
        self.hat_c_o = make_C(self.o, self.o_g, name="s")

        with tf.name_scope("train"):
            self.cost_loss = tf.losses.mean_squared_error()
            self.loss = self.cost_loss + self.state_prediction_loss + self.cost_prediction_loss
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, global_step=self.global_step)
            trainable_vars = tf.trainable_variables()
            grads = zip(tf.gradients(self.loss, trainable_vars), trainable_vars)
            for grad, var in grads:
                tf.summary.histogram(var.name + "/gradient", grad)

        tf.summary.scalar("loss", self.loss)

    def reduce(self, s):
        feed_dict = {self.x: s}
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

    def train(self, x_train, y_train, epochs=5, checkpoint=None):
        """
        x train is an array, each row of which looks like:
            [s_t, u_t, s_{t+1}, goal]
        y train is an array, each row of which looks like:
            [c_t, c_{t+1}]
         """
        # load our train training data and train to it
        if checkpoint:
            self.saver.restore(self.sess, checkpoint)
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

        batch_size = 32
        for i in range(epochs):
            start = np.random.randint(0, self.n_training_samples - batch_size)
            batch_x = self.train_x[start, start + batch_size][0, 1, 2]
            batch_y = self.train_y[start, start + batch_size]
            feed_dict = {self.x: batch_x, self.y: batch_y}
            ops = [self.global_step, self.summaries, self.loss]
            step, summary, loss = self.sess.run(ops, feed_dict=feed_dict)
            self.writer.add_summary(summary, step)

        self.saver.save(self.sess, os.path.join(self.log_dir, "nn.ckpt"))


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", '-m', default="myfirst")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("train_data", help="train training data file")
    parser.add_argument("model_save_file", help="train training data file")
    parser.add_argument("outfile", help='filename to store data in')
    parser.add_argument("--load", help="load this saved model file")
    parser.add_argument("--epochs", '-e', type=int, default=10, help="number of epochs to train for")

    args = parser.parse_args()

    n, x, y = load_and_construct_training_data(args.train_data)

    model = NNModel(n)
    model.train(x, y)
