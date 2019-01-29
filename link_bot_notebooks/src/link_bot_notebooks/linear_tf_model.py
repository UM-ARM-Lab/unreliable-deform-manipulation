#!/usr/bin/env python
from __future__ import print_function

import os
import json

import numpy as np
import tensorflow as tf
from colorama import Fore
from link_bot_notebooks import base_model
from link_bot_notebooks import toy_problem_optimization_common as tpo


class LinearTFModel(base_model.BaseModel):

    def __init__(self, args, batch_size, N, M, L, dt, n_steps, seed=0):
        base_model.BaseModel.__init__(self, N, M, L)

        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        self.batch_size = batch_size
        self.args = args
        self.N = N
        self.M = M
        self.L = L
        self.beta = 1e-8
        self.n_steps = n_steps
        self.dt = dt

        self.s = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1, N), name="s")
        self.u = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps, L), name="u")
        self.g = tf.placeholder(tf.float32, shape=(1, N), name="g")
        self.c = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1), name="c")

        self.A = tf.get_variable("A", shape=[N, M])
        self.B = tf.get_variable("B", shape=[M, M], initializer=tf.zeros_initializer())
        self.C = tf.get_variable("C", shape=[M, L], initializer=tf.zeros_initializer())

        # we force D to be identity because it's tricky to constrain it to be positive semi-definite
        self.D = tf.Variable(np.eye(self.M, dtype=np.float32), trainable=False)

        self.hat_o = tf.einsum('bsn,nm->bsm', self.s, self.A, name='reduce')
        self.og = tf.matmul(self.g, self.A, name='reduce_goal')

        # by copying hat_o we are setting the first element of hat_o_ correctly
        self.hat_o_ = tf.get_variable("hat_o_", shape=[batch_size, self.n_steps + 1, M])
        for i in range(1, self.n_steps + 1):
            Bo = tf.einsum('mp,bp->bm', self.dt * self.B, self.hat_o_[:, i - 1], name='Bo')
            Cu = tf.einsum('ml,bl->bm', self.dt * self.C, self.u[:, i - 1], name='Cu')
            oi = self.hat_o_[:, i - 1] + Bo + Cu
            tf.assign(self.hat_o_[:, i], oi)

        self.d_to_goal = self.og - self.hat_o
        self.hat_c = tf.einsum('bst,tp,bsp->bs', self.d_to_goal, self.D, self.d_to_goal)

        with tf.name_scope("train"):
            self.cost_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            self.state_prediction_error = tf.norm(self.hat_o - self.hat_o_, axis=0)
            self.state_prediction_loss = tf.reduce_mean(self.state_prediction_error)
            self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            flat_weights = tf.concat((tf.reshape(self.A, [-1]), tf.reshape(self.B, [-1]),
                                      tf.reshape(self.C, [-1]), tf.reshape(self.D, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(flat_weights) * self.beta
            self.loss = self.cost_loss + self.state_prediction_loss + self.cost_prediction_loss + self.regularization
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.opt = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

            trainable_vars = tf.trainable_variables()
            grads = zip(tf.gradients(self.loss, trainable_vars), trainable_vars)
            for grad, var in grads:
                name = var.name.replace(":", "_")
                tf.summary.histogram(name + "/gradient", grad)

            tf.summary.scalar("cost_loss", self.cost_loss)
            tf.summary.scalar("state_prediction_loss", self.state_prediction_loss)
            tf.summary.scalar("cost_prediction_loss", self.cost_prediction_loss)
            tf.summary.scalar("regularization_loss", self.regularization)
            tf.summary.scalar("loss", self.loss)

            self.summaries = tf.summary.merge_all()
            self.sess = tf.Session()
            self.saver = tf.train.Saver(max_to_keep=None)

    def train(self, train_x, goal, epochs, log_path):
        interrupted = False

        writer = None
        loss = None
        full_log_path = None
        if self.args['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)

            tpo.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "metadata.json")
            metadata_file = open(metadata_path, 'w')
            metadata = {
                'log path': full_log_path,
                'checkpoint': self.args['checkpoint'],
                'N': self.N,
                'M': self.M,
                'L': self.L,
                'beta': self.beta,
                'dt': self.dt,
                'commandline': self.args['commandline'],
            }
            metadata_file.write(json.dumps(metadata, indent=2))

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
                    if self.args['log'] is not None:
                        writer.add_summary(summary, step)
                        self.save(full_log_path, loss=loss)
                    else:
                        print(step, loss)
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

    def predict_cost(self, o, u, g):
        feed_dict = {self.hat_o: o, self.u: u, self.g: g}
        ops = [self.hat_c_]
        hat_c_ = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_c_

    def predict_from_o(self, o, u):
        return self.predict(o, u)

    def cost_of_s(self, s, g):
        return self.cost(self.reduce(s), g)

    def cost(self, o, g):
        feed_dict = {self.hat_o: o, self.g: g}
        ops = [self.hat_c]
        hat_c = self.sess.run(ops, feed_dict=feed_dict)[0]
        hat_c = np.expand_dims(hat_c, axis=0)
        return hat_c

    def save(self, log_path, log=True, loss=None):
        global_step = self.sess.run(self.global_step)
        if log:
            if loss:
                print(Fore.CYAN + "Saving ckpt {} at step {:d} with loss {}".format(log_path, global_step,
                                                                                    loss) + Fore.RESET)
            else:
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

            # visualize a few sample predictions from the testing data
            self.sess.run([self.hat_o_], feed_dict=feed_dict)

        return A, B, C, D, c_loss, sp_loss, cp_loss, reg, loss

    def batch(self, x, goal):
        """ x is 3d.
            first axis is the trajectory.
            second axis is the time step
            third axis is the state/action data
        """
        batch_size = min(x.shape[2], self.args['batch_size'])
        if batch_size == x.shape[2]:
            batch_examples = x
        else:
            batch_indeces = np.random.randint(0, x.shape[2], size=batch_size)
            batch_examples = x[:, :, batch_indeces]

        # there is always only one s and one s_, but the amount of time between them can vary but changing
        # the length of the trajectories loaded for training, via the parameter 'trajectory_length_to_train'
        print(batch_examples.shape)
        for i in range(batch_examples.shape[0]):
            s = batch_examples[:self.n_steps, :self.N, :]
            u = batch_examples[:self.n_steps, self.N:, :]

        # Here we compute the label for cost/reward and constraints
        c = np.sum((s[:, [0, 1], :] - goal[[0, 1]]) ** 2, axis=1)

        return s, s_, u, c, c_

    def get_ABCD(self):
        feed_dict = {}
        ops = [self.A, self.B, self.C, self.D]
        return self.sess.run(ops, feed_dict=feed_dict)

    def get_A(self):
        feed_dict = {}
        ops = [self.A]
        A = self.sess.run(ops, feed_dict=feed_dict)[0]
        return A

    def __str__(self):
        A, B, C, D = self.get_ABCD()
        return "A:\n" + np.array2string(A) + "\n" + \
               "B:\n" + np.array2string(B) + "\n" + \
               "C:\n" + np.array2string(C) + "\n" + \
               "D:\n" + np.array2string(D) + "\n"
