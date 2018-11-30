#!/usr/bin/env python
from __future__ import print_function

import os
from colorama import Fore
import numpy as np
import tensorflow as tf


from link_bot_notebooks import base_model


class LinearTFModel(base_model.BaseModel):

    def __init__(self, args, N, M, L, seed=0):
        base_model.BaseModel.__init__(self, N, M, L)

        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        self.args = args
        self.N = N
        self.M = M
        self.L = L
        self.beta = 1e-8

        self.s = tf.placeholder(tf.float32, shape=(N, None), name="s")
        self.s_ = tf.placeholder(tf.float32, shape=(N, None), name="s_")
        self.u = tf.placeholder(tf.float32, shape=(L, None), name="u")
        self.g = tf.placeholder(tf.float32, shape=(N, None), name="g")
        self.c = tf.placeholder(tf.float32, shape=(None), name="c")
        self.c_ = tf.placeholder(tf.float32, shape=(None), name="c_")

        self.A = tf.Variable(tf.truncated_normal([M, N]), name="A")
        self.B = tf.Variable(tf.truncated_normal([M, M]), name="B")
        self.C = tf.Variable(tf.truncated_normal([M, L]), name="C")
        self.D = tf.Variable(tf.truncated_normal([M, M]), name="D")

        self.hat_o = tf.matmul(self.A, self.s, name='reduce')
        self.og = tf.matmul(self.A, self.g, name='reduce_goal')
        self.o_ = tf.matmul(self.A, self.s_, name='reduce_')
        self.hat_o_ = self.hat_o + tf.matmul(self.B, self.hat_o, name='dynamics') + \
                      tf.matmul(self.C, self.u, name='controls')
        self.d_to_goal = self.og - self.hat_o
        self.d_to_goal_ = self.og - self.hat_o_
        self.hat_c = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal), self.D), self.d_to_goal))
        self.hat_c_ = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal_), self.D), self.d_to_goal_))

        with tf.name_scope("train"):
            self.cost_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            self.state_prediction_loss = tf.losses.mean_squared_error(labels=self.o_, predictions=self.hat_o_)
            self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c_, predictions=self.hat_c_)
            flat_weights = tf.concat((tf.reshape(self.A, [-1]), tf.reshape(self.B, [-1]),
                                      tf.reshape(self.C, [-1]), tf.reshape(self.D, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(flat_weights) * self.beta
            self.loss = self.cost_loss + self.state_prediction_loss + self.cost_prediction_loss + self.regularization
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            starter_learning_rate = 0.01
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

    def train(self, train_x, train_y, epochs, log_path):
        """
        x train is an array, each row of which looks like:
            [s_t, u_t, s_{t+1}, goal]
        y train is an array, each row of which looks like:
            [c_t, c_{t+1}]
        """
        interrupted = False
        n_training_samples = train_x.shape[0]
        batch_size = self.args['batch_size']

        if self.args['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)
            writer = tf.summary.FileWriter(full_log_path)
            writer.add_graph(self.sess.graph)

        try:
            if self.args['verbose']:
                print("TRAINING FOR {} EPOCHS:".format(epochs))
            for i in range(epochs):
                if batch_size == -1:
                    start = 0
                    end = n_training_samples
                else:
                    start = np.random.randint(0, n_training_samples - batch_size)
                    end = start + batch_size
                batch_s = train_x[start:end, 0:self.N].T
                batch_s_ = train_x[start:end, self.N: 2 * self.N].T
                batch_g = train_x[start:end, 2 * self.N: 3 * self.N].T
                batch_u = train_x[start:end, 3 * self.N: 3 * self.N + self.L].T
                batch_c = train_y[start:end, 0]
                batch_c_ = train_y[start:end, 1]
                feed_dict = {self.s: batch_s,
                             self.s_: batch_s_,
                             self.u: batch_u,
                             self.g: batch_g,
                             self.c: batch_c,
                             self.c_: batch_c_}
                ops = [self.global_step, self.summaries, self.loss, self.opt]
                step, summary, loss, _ = self.sess.run(ops, feed_dict=feed_dict)

                if step % self.args['print_period'] == 0:
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

    def act(self, o, g):
        """ return the action which gives the lowest cost for the predicted next state """
        feed_dict = {self.hat_o: o, self.g: g}
        ops = [self.B, self.C, self.og]
        B, C, og = self.sess.run(ops, feed_dict=feed_dict)
        u = np.linalg.lstsq(C, (og - o - np.dot(B, o)), rcond=None)[0]

        feed_dict = {self.hat_o: o, self.g: g, self.u: u}
        ops = [self.hat_o_, self.hat_c_]
        hat_o_, hat_c_ = self.sess.run(ops, feed_dict=feed_dict)
        return u, hat_c_, hat_o_

    def save(self, log_path):
        global_step = self.sess.run(self.global_step)
        print(Fore.CYAN + "Saving ckpt {} at step {:d}".format(log_path, global_step) + Fore.RESET)
        self.saver.save(self.sess, os.path.join(log_path, "nn.ckpt"), global_step=self.global_step)

    def load(self):
        self.saver.restore(self.sess, self.args['checkpoint'])
        global_step = self.sess.run(self.global_step)
        print(Fore.CYAN + "Restored ckpt {} at step {:d}".format(self.args['checkpoint'], global_step) + Fore.RESET)

    def evaluate(self, x_eval, y_eval, display=True):
        s = x_eval[:, 0:self.N].T
        s_ = x_eval[:, self.N: 2 * self.N].T
        g = x_eval[:, 2 * self.N: 3 * self.N].T
        u = x_eval[:, 3 * self.N: 3 * self.N + self.L].T
        c = y_eval[:, 0]
        c_ = y_eval[:, 1]
        feed_dict = {self.s: s,
                     self.s_: s_,
                     self.u: u,
                     self.g: g,
                     self.c: c,
                     self.c_: c_}
        ops = [self.A, self.B, self.C, self.D, self.cost_loss, self.state_prediction_loss, self.cost_prediction_loss,
               self.regularization, self.loss]
        A, B, C, D, c_loss, sp_loss, cp_loss, reg, loss = self.sess.run(ops, feed_dict=feed_dict)
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
