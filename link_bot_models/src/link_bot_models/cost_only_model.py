#!/usr/bin/env python
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from link_bot_models import base_model
from tensorflow.python import debug as tf_debug


class CostOnlyModel(base_model.BaseModel):

    def __init__(self, args, N, M, L, n_steps, dt, seed=0):
        super(CostOnlyModel, self).__init__(args, N)

        self.M = M
        self.L = L
        self.n_steps = n_steps
        self.beta = 1e-16
        self.dt = dt

        self.s = tf.placeholder(tf.float32, shape=(N, None), name="s")
        self.s_ = tf.placeholder(tf.float32, shape=(N, None), name="s_")
        self.u = tf.placeholder(tf.float32, shape=(n_steps, L, None), name="u")
        self.g = tf.placeholder(tf.float32, shape=(N, 1), name="g")
        self.c = tf.placeholder(tf.float32, shape=(None), name="c")
        self.c_ = tf.placeholder(tf.float32, shape=(None), name="c_")

        self.A = tf.Variable(tf.truncated_normal(shape=[M, N]), name="A", dtype=tf.float32)
        self.D = tf.Variable(tf.truncated_normal(shape=[M, M]), name="D", dtype=tf.float32)

        self.hat_o = tf.matmul(self.A, self.s, name='reduce')
        self.og = tf.matmul(self.A, self.g, name='reduce_goal')

        self.d_to_goal = self.og - self.hat_o
        self.hat_c = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal), self.D), self.d_to_goal))

        with tf.name_scope("train"):
            self.cost_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            flat_weights = tf.concat((tf.reshape(self.A, [-1]), tf.reshape(self.D, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(flat_weights) * self.beta
            self.loss = self.cost_loss + self.regularization
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            starter_learning_rate = 0.001
            self.opt = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

            trainable_vars = tf.trainable_variables()
            grads = zip(tf.gradients(self.loss, trainable_vars), trainable_vars)
            for grad, var in grads:
                name = var.name.replace(":", "_")
                tf.summary.histogram(name + "/gradient", grad)

            tf.summary.scalar("cost_loss", self.cost_loss)
            tf.summary.scalar("regularization_loss", self.regularization)
            tf.summary.scalar("loss", self.loss)

            # Set up logging/saving
            self.summaries = tf.summary.merge_all()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.015)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            if 'debug' in self.args and self.args['debug']:
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
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
            s, s_, u, c, c_ = self.batch(train_x, goal)
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
            ops = [self.A, self.D]
            A, D = self.sess.run(ops, feed_dict={})
            if self.args['verbose']:
                print("Loss: {}".format(loss))
                print("A:\n{}".format(A))
                print("D:\n{}".format(D))

            if self.args['log'] is not None:
                self.save(full_log_path)

        return interrupted

    def reduce(self, s):
        feed_dict = {self.s: s}
        ops = [self.hat_o]
        hat_o = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o

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

    def evaluate(self, eval_x, goal, display=True):
        s, s_, u, c, c_ = self.batch(eval_x, goal)
        feed_dict = {self.s: s,
                     self.s_: s_,
                     self.u: u,
                     self.g: goal,
                     self.c: c,
                     self.c_: c_}
        ops = [self.A,  self.D, self.cost_loss, self.regularization, self.loss]
        A, D, c_loss, reg, loss = self.sess.run(ops, feed_dict=feed_dict)
        if display:
            print("Cost Loss: {}".format(c_loss))
            print("Regularization: {}".format(reg))
            print("Overall Loss: {}".format(loss))
            print("A:\n{}".format(A))
            print("D:\n{}".format(D))
        return A, D, c_loss, reg, loss

    def batch(self, x, goal):
        """ x is 3d.
            first axis is the time step
            second axis is the state/action data
            third axis is the trajectory.
        """
        if x.shape[0] < self.n_steps:
            raise Exception("Need more time steps of data!")

        batch_size = min(x.shape[2], self.args['batch_size'])
        example_indeces = np.arange(x.shape[2])
        np.random.shuffle(example_indeces)
        batch_indeces = example_indeces[:batch_size]
        s = x[0, :self.N, :][:, batch_indeces]
        s_ = x[self.n_steps, :self.N, :][:, batch_indeces]
        u = x[:-1, self.N:, batch_indeces]
        c = np.sum((x[0, [0, 1]][:, batch_indeces] - goal[[0, 1]]) ** 2, axis=0)
        c_ = np.sum((x[-1, [0, 1]][:, batch_indeces] - goal[[0, 1]]) ** 2, axis=0)
        return s, s_, u, c, c_

    def get_AD(self):
        feed_dict = {}
        ops = [self.A, self.D]
        return self.sess.run(ops, feed_dict=feed_dict)

    def get_A(self):
        feed_dict = {}
        ops = [self.A]
        A = self.sess.run(ops, feed_dict=feed_dict)[0]
        return A

    def __str__(self):
        A, D = self.get_AD()
        return "A:\n" + np.array2string(A) + "\n" + \
               "D:\n" + np.array2string(D) + "\n"
