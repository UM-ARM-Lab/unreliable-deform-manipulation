#!/usr/bin/env python
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from colorama import Fore
from link_bot_notebooks import base_model
from tensorflow.python import debug as tf_debug


class LinearInvertibleModel(base_model.BaseModel):

    def __init__(self, args, N, M, K, L, dt, seed=0):
        base_model.BaseModel.__init__(self, N, M, L)

        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        self.args = args
        self.N = N
        self.M = M
        self.K = K
        self.L = L
        self.beta = 1e-8
        self.dt = dt

        self.s = tf.placeholder(tf.float32, shape=(N, None), name="s")
        self.s_ = tf.placeholder(tf.float32, shape=(N, None), name="s_")
        self.u = tf.placeholder(tf.float32, shape=(1, L, None), name="u")
        self.g = tf.placeholder(tf.float32, shape=(N, 1), name="g")
        self.c = tf.placeholder(tf.float32, shape=(None), name="c")
        self.c_ = tf.placeholder(tf.float32, shape=(None), name="c_")

        self.A_control = tf.Variable(tf.truncated_normal(shape=[M, N]), name="A_control", dtype=tf.float32)
        self.A_constraint = tf.Variable(tf.truncated_normal(shape=[K, M]), name="A_constraint", dtype=tf.float32)
        self.B = tf.Variable(tf.truncated_normal(shape=[M, M], stddev=1e-2), name="B", dtype=tf.float32)
        self.C = tf.Variable(tf.truncated_normal(shape=[M, L]), name="C", dtype=tf.float32)
        self.B_inv = tf.Variable(tf.truncated_normal(shape=[L, M]), name="B_inv", dtype=tf.float32)
        self.C_inv = tf.Variable(tf.truncated_normal(shape=[L, M]), name="C_inv", dtype=tf.float32)
        self.D = np.eye(M, dtype=np.float32)

        self.hat_o_control = tf.matmul(self.A_control, self.s, name='reduce')
        self.hat_o_constraint = tf.matmul(self.A_constraint, self.hat_o_control, name='constraint')
        self.og = tf.matmul(self.A_control, self.g, name='reduce_goal')
        self.o_control_ = tf.matmul(self.A_control, self.s_, name='reduce_')

        self.state_bo = tf.matmul(self.B, self.hat_o_control, name='dynamics')
        self.state_o_ = self.hat_o_control + self.state_bo
        self.temp_o_control = tf.matmul(self.dt * self.C, self.u[0], name='controls')
        self.hat_o_control_ = tf.add(self.state_o_, self.temp_o_control, name='hat_o_')

        self.d_to_goal = self.og - self.hat_o_control
        self.d_to_goal_ = self.og - self.hat_o_control_
        self.hat_c = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal), self.D), self.d_to_goal))
        self.hat_c_ = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal_), self.D), self.d_to_goal_))

        self.hat_u = 1.0 / self.dt * (
                    tf.matmul(self.B_inv, self.hat_o_control) + tf.matmul(self.C_inv, self.hat_o_control_))

        with tf.name_scope("train"):
            self.cost_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            self.state_prediction_loss = tf.reduce_mean(tf.norm(self.o_control_ - self.hat_o_control_, axis=0))
            self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c_, predictions=self.hat_c_)
            flat_weights = tf.concat((tf.reshape(self.A_control, [-1]), tf.reshape(self.B, [-1]),
                                      tf.reshape(self.C, [-1]), tf.reshape(self.D, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(flat_weights) * self.beta
            self.loss = self.cost_loss + self.state_prediction_loss + self.cost_prediction_loss + self.regularization
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.opt = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

            trainable_vars = tf.trainable_variables()
            for var in trainable_vars:
                grads = tf.gradients(self.loss, var)
                for grad in grads:
                    if grad is not None:
                        name = var.name.replace(":", "_")
                        tf.summary.histogram(name + "/gradient", grad)
                    else:
                        print("Warning... there is no gradient of the loss with respect to {}".format(var.name))

            tf.summary.scalar("cost_loss", self.cost_loss)
            tf.summary.scalar("state_prediction_loss", self.state_prediction_loss)
            tf.summary.scalar("cost_prediction_loss", self.cost_prediction_loss)
            tf.summary.scalar("regularization_loss", self.regularization)
            tf.summary.scalar("loss", self.loss)

            self.summaries = tf.summary.merge_all()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.015)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            if 'tf-debug' in self.args and self.args['tf-debug']:
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.saver = tf.train.Saver(max_to_keep=None)

    def train(self, train_x, goal, epochs, log_path):
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
            ops = [self.global_step, self.summaries, self.loss, self.opt, self.B]
            for i in range(epochs):
                step, summary, loss, _, B = self.sess.run(ops, feed_dict=feed_dict)

                if 'print_period' in self.args and step % self.args['print_period'] == 0:
                    print(step, loss)

                if self.args['log'] is not None:
                    writer.add_summary(summary, step)
        except KeyboardInterrupt:
            print("stop!!!")
            interrupted = True
            pass
        finally:
            ops = [self.A_control, self.B, self.C, self.D]
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
        ops = [self.hat_o_control]
        hat_o_control = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_control

    def constraint(self, o):
        feed_dict = {self.hat_o_controlo: o}
        ops = [self.hat_o_constraint]
        hat_o_constraint = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_constraint

    def inverse(self, o, o_):
        feed_dict = {self.hat_o_control: o, self.hat_o_control_: o_}
        ops = [self.hat_u]
        hat_u = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_u

    def predict(self, o, u):
        feed_dict = {self.hat_o_control: o, self.u: u}
        ops = [self.hat_o_control_]
        hat_o_ = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_

    def predict_from_o(self, o, u):
        return self.predict(o, u)

    def cost_of_s(self, s, g):
        return self.cost(self.reduce(s), g)

    def cost(self, o, g):
        feed_dict = {self.hat_o_control: o, self.g: g}
        ops = [self.hat_c]
        hat_c = self.sess.run(ops, feed_dict=feed_dict)[0]
        hat_c = np.expand_dims(hat_c, axis=0)
        return hat_c

    def save(self, log_path):
        global_step = self.sess.run(self.global_step)
        print(Fore.CYAN + "Saving ckpt {} at step {:d}".format(log_path, global_step) + Fore.RESET)
        self.saver.save(self.sess, os.path.join(log_path, "nn.ckpt"), global_step=self.global_step)

    def load(self):
        self.saver.restore(self.sess, self.args['checkpoint'])
        global_step = self.sess.run(self.global_step)
        print(Fore.CYAN + "Restored ckpt {} at step {:d}".format(self.args['checkpoint'], global_step) + Fore.RESET)

    def evaluate(self, eval_x, goal, display=True):
        s, s_, u, c, c_ = self.batch(eval_x, goal)
        feed_dict = {self.s: s,
                     self.s_: s_,
                     self.u: u,
                     self.g: goal,
                     self.c: c,
                     self.c_: c_}
        ops = [self.A_control, self.B, self.C, self.D, self.cost_loss, self.state_prediction_loss,
               self.cost_prediction_loss,
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

    def batch(self, x, goal):
        batch_size = min(x.shape[2], self.args['batch_size'])
        example_indeces = np.arange(x.shape[2])
        batch_indeces = example_indeces[:batch_size]
        s = x[0, :self.N, :][:, batch_indeces]
        s_ = x[1, :self.N, :][:, batch_indeces]
        u = x[:-1, self.N:, batch_indeces]
        c = np.sum((x[0, [0, 1]][:, batch_indeces] - goal[[0, 1]]) ** 2, axis=0)
        c_ = np.sum((x[-1, [0, 1]][:, batch_indeces] - goal[[0, 1]]) ** 2, axis=0)
        return s, s_, u, c, c_

    def get_ABCD(self):
        feed_dict = {}
        ops = [self.A_control, self.B, self.C, self.D]
        return self.sess.run(ops, feed_dict=feed_dict)

    def get_A(self):
        feed_dict = {}
        ops = [self.A_control]
        A = self.sess.run(ops, feed_dict=feed_dict)[0]
        return A

    def __str__(self):
        A, B, C, D = self.get_ABCD()
        return "A:\n" + np.array2string(A) + "\n" + \
               "B:\n" + np.array2string(B) + "\n" + \
               "C:\n" + np.array2string(C) + "\n" + \
               "D:\n" + np.array2string(D) + "\n"
