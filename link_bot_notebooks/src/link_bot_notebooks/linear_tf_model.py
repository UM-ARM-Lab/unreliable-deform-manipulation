#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import os
import json

import numpy as np
import tensorflow as tf
from colorama import Fore
from link_bot_notebooks import base_model
from link_bot_notebooks import toy_problem_optimization_common as tpo
from tensorflow.python import debug as tf_debug

true_fake_B = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
true_fake_C = np.array([[2, 1], [0, 3]], dtype=np.float32)


class LinearTFModel(base_model.BaseModel):

    def __init__(self, args, batch_size, N, M, L, dt, n_steps, seed=0):
        base_model.BaseModel.__init__(self, N, M, L)

        self.seed = seed
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)

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

        # self.A = tf.get_variable("A", [N, M], initializer=tf.initializers.truncated_normal(0, 1, seed=self.seed))
        # self.B = tf.get_variable("B", [M, M], initializer=tf.initializers.truncated_normal(0, 0.01, seed=self.seed))
        # self.C = tf.get_variable("C", [M, L], initializer=tf.initializers.truncated_normal(0, 1, seed=self.seed))

        self.A = tf.get_variable("A", initializer=np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float32))
        self.B = tf.get_variable("B", initializer=true_fake_B)
        self.C = tf.get_variable("C", initializer=true_fake_C)

        # we force D to be identity because it's tricky to constrain it to be positive semi-definite
        self.D = tf.Variable(np.eye(self.M, dtype=np.float32), trainable=False, name="D")

        self.hat_o = tf.einsum('bsn,nm->bsm', self.s, self.A, name='hat_o')
        self.og = tf.matmul(self.g, self.A, name='reduce_goal')

        hat_o_next = []
        hat_o_next.append(self.hat_o[:, 0, :])

        for i in range(1, self.n_steps + 1):
            Bo = tf.einsum('mp,bp->bm', self.dt * self.B, hat_o_next[i - 1], name='Bo')
            Cu = tf.einsum('ml,bl->bm', self.dt * self.C, self.u[:, i - 1], name='Cu')
            hat_o_next.append(hat_o_next[i - 1] + Bo + Cu)

        self.hat_o_next = tf.transpose(tf.stack(hat_o_next), [1, 0, 2], name='hat_o_next')

        self.d_to_goal = self.og - self.hat_o_next
        self.hat_c = tf.einsum('bst,tp,bsp->bs', self.d_to_goal, self.D, self.d_to_goal)

        with tf.name_scope("train"):
            # Euclidean error in latent space at each time step
            state_prediction_error = tf.reduce_sum(tf.pow(self.hat_o - self.hat_o_next, 2), axis=2)
            self.state_prediction_loss = tf.reduce_mean(state_prediction_error)
            self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            self.flat_weights = tf.concat(
                (tf.reshape(self.A, [-1]), tf.reshape(self.B, [-1]), tf.reshape(self.C, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(self.flat_weights) * self.beta

            self.loss = tf.add_n([self.state_prediction_loss, self.cost_prediction_loss, self.regularization],
                                 name='loss')

            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.opt = tf.train.AdamOptimizer(learning_rate=0.002).minimize(self.loss, global_step=self.global_step)

            trainable_vars = tf.trainable_variables()
            for var in trainable_vars:
                name = var.name.replace(":", "_")
                grads = tf.gradients(self.loss, var, name='dLoss_d{}'.format(name))
                for grad in grads:
                    if grad is not None:
                        tf.summary.histogram(name + "/gradient", grad)
                    else:
                        print("Warning... there is no gradient of the loss with respect to {}".format(var.name))

            tf.summary.scalar("state_prediction_loss", self.state_prediction_loss)
            tf.summary.scalar("cost_prediction_loss", self.cost_prediction_loss)
            tf.summary.scalar("regularization_loss", self.regularization)
            tf.summary.scalar("loss", self.loss)

            self.summaries = tf.summary.merge_all()
            self.sess = tf.Session()
            if args['debug']:
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
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
                'tf_version': str(tf.__version__),
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
            s, u, c = self.compute_cost_label(train_x, goal)
            feed_dict = {self.s: s,
                         self.u: u,
                         self.g: goal,
                         self.c: c}
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

    def evaluate(self, eval_x, goal, display=True):
        s, u, c = self.compute_cost_label(eval_x, goal)
        feed_dict = {self.s: s,
                     self.u: u,
                     self.g: goal,
                     self.c: c}
        ops = [self.A, self.B, self.C, self.D, self.state_prediction_loss, self.cost_prediction_loss,
               self.regularization, self.loss]
        A, B, C, D, c_loss, sp_loss, reg, loss = self.sess.run(ops, feed_dict=feed_dict)
        if display:
            print("Cost Loss: {}".format(c_loss))
            print("State Prediction Loss: {}".format(sp_loss))
            print("Regularization: {}".format(reg))
            print("Overall Loss: {}".format(loss))
            print("A:\n{}".format(A))
            print("B:\n{}".format(B))
            print("C:\n{}".format(C))
            print("D:\n{}".format(D))

            # visualize a few sample predictions from the testing data
            self.sess.run([self.hat_o_next], feed_dict=feed_dict)

        return A, B, C, D, c_loss, sp_loss, reg, loss

    def compute_cost_label(self, x, goal):
        """ x is 3d.
            first axis is the trajectory.
            second axis is the time step
            third axis is the [state|action] data
        """
        s = x[:, :, 1:self.N + 1]
        u = x[:, :-1, self.N + 1:]
        # NOTE: Here we compute the label for cost/reward and constraints
        c = np.sum((s[:, :, [0, 1]] - goal[0, [0, 1]]) ** 2, axis=2)
        return s, u, c

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
        ops = [self.hat_o_next]
        hat_o_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_next

    def predict_cost(self, o, u, g):
        feed_dict = {self.hat_o: o, self.u: u, self.g: g}
        ops = [self.hat_c_]
        hat_c_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_c_next

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
