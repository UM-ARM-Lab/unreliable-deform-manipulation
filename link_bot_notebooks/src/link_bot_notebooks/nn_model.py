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

    def __init__(self, args, N, M, L, n_steps, dt, seed=0):
        base_model.BaseModel.__init__(self, N, M, L)

        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        self.args = args
        self.N = N
        self.M = M
        self.L = L
        self.n_steps = n_steps
        self.beta = 1e-8
        self.dt = dt

        self.s = tf.placeholder(tf.float32, shape=(N, None), name="s")
        self.s_ = tf.placeholder(tf.float32, shape=(N, None), name="s_")
        self.u = tf.placeholder(tf.float32, shape=(n_steps, L, None), name="u")
        self.g = tf.placeholder(tf.float32, shape=(N, 1), name="g")
        self.c = tf.placeholder(tf.float32, shape=(None), name="c")
        self.c_ = tf.placeholder(tf.float32, shape=(None), name="c_")

        h1 = 64
        h2 = 32
        self.W1 = tf.get_variable("W1", shape=[M, h1])
        self.B1 = tf.get_variable("B1", shape=[h1])
        self.W2 = tf.get_variable("W2", shape=[h1, h2])
        self.B2 = tf.get_variable("B2", shape=[h2])
        self.W3 = tf.get_variable("W3", shape=[h2, M])
        self.B3 = tf.get_variable("B3", shape=[M])

        # we force D to be identity because it's tricky to constrain it to be positive semi-definit and also learned
        self.D = tf.get_variable("D", initializer=np.eye(self.M, dtype=np.float32), trainable=False)

        def reduce(s, name):
            with tf.name_scope("reduce_" + name):
                a1 = tf.nn.relu(tf.matmul(s, self.W1) + self.B1)
                a2 = tf.nn.relu(tf.matmul(a1, self.W2) + self.B2)
                o = tf.matmul(a2, self.W3) + self.B3
            return o

        self.hat_o = reduce(self.s, 'hat_o')
        self.og = reduce(self.g, 'reduce_goal')
        self.o_ = reduce(self.s_, 'reduce_')

        self.state_bo = tf.matmul(self.dt * self.B, self.hat_o, name='dynamics'.format(0))
        self.state_o_ = self.hat_o + self.state_bo
        self.control_o_ = tf.matmul(self.dt * self.C, self.u[0], name='controls'.format(0))
        self.hat_o_ = tf.add(self.state_o_, self.control_o_, name='hat_o_')
        for i in range(self.n_steps - 1):
            with tf.name_scope("step_{}".format(i)):
                self.state_o_ = self.hat_o_ + tf.matmul(self.dt * self.B, self.hat_o_, name='dynamics'.format(i))
                self.control_o_ = tf.matmul(self.dt * self.C, self.u[i], name='controls'.format(i))
                self.hat_o_ = tf.add(self.state_o_, self.control_o_, name='hat_o_')

        self.d_to_goal = self.og - self.hat_o
        self.d_to_goal_ = self.og - self.hat_o_
        self.hat_c = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal), self.D), self.d_to_goal))
        self.hat_c_ = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal_), self.D), self.d_to_goal_))

        with tf.name_scope("train"):
            self.cost_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            self.state_prediction_error = tf.norm(self.o_ - self.hat_o_, axis=0)
            self.state_prediction_loss = tf.reduce_mean(self.state_prediction_error)
            self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c_, predictions=self.hat_c_)
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
                'n_steps': self.n_steps,
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
                print(Fore.CYAN + "Saving ckpt {} at step {:d} with loss {}".format(log_path, global_step, loss) + Fore.RESET)
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
            first axis is the time step
            second axis is the state/action data
            third axis is the trajectory.
        """
        batch_size = min(x.shape[2], self.args['batch_size'])
        if batch_size == x.shape[2]:
            batch_examples = x
        else:
            batch_indeces = np.random.randint(0, x.shape[2], size=batch_size)
            batch_examples = x[:, :, batch_indeces]

        # there is always only one s and one s_, but the amount of time between them can vary but changing
        # the length of the trajectories loaded for training, via the parameter 'trajectory_length_to_train'
        s = batch_examples[0, :self.N, :]
        s_ = batch_examples[-1, :self.N, :]
        u = batch_examples[0:self.n_steps, self.N:, :]

        # Here we compute the label for cost/reward and constraints
        c = np.sum((s[[0, 1], :] - goal[[0, 1]]) ** 2, axis=0)
        c_ = np.sum((s_[[0, 1], :] - goal[[0, 1]]) ** 2, axis=0)

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
