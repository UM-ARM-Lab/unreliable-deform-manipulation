#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import control
import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_models import base_model


class LinearTFModel(base_model.BaseModel):

    def __init__(self, args, batch_size, N, M, L, dt, n_steps):
        super(LinearTFModel, self).__init__(args, N)

        self.batch_size = batch_size
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
        # self.B = tf.get_variable("B", [M, M], initializer=tf.initializers.truncated_normal(0, 0.1, seed=self.seed))
        # self.C = tf.get_variable("C", [M, L], initializer=np.eye)

        a_init = np.random.randn(N, M).astype(np.float32) * 1e-6
        a_init[0, 0] = 1
        a_init[1, 1] = 1
        b_init = np.random.randn(M, M).astype(np.float32) * 1e-6
        c_init = np.random.randn(M, L).astype(np.float32) * 1e-6
        np.fill_diagonal(c_init, 1)
        self.A = tf.get_variable("A", initializer=a_init)
        self.B = tf.get_variable("B", initializer=b_init)
        self.C = tf.get_variable("C", initializer=c_init)

        # we force D to be identity because it's tricky to constrain it to be positive semi-definite
        self.D = tf.get_variable("D", initializer=np.eye(self.M, dtype=np.float32), trainable=False)

        self.hat_o = tf.einsum('bsn,nm->bsm', self.s, self.A, name='hat_o')
        self.og = tf.matmul(self.g, self.A, name='reduce_goal')

        hat_o_next = [self.hat_o[:, 0, :]]

        for i in range(1, self.n_steps + 1):
            Bo = tf.einsum('mp,bp->bm', self.dt * self.B, hat_o_next[i - 1], name='Bo')
            Cu = tf.einsum('ml,bl->bm', self.dt * self.C, self.u[:, i - 1], name='Cu')
            hat_o_next.append(hat_o_next[i - 1] + Bo + Cu)

        self.hat_o_next = tf.transpose(tf.stack(hat_o_next), [1, 0, 2], name='hat_o_next')

        self.d_to_goal = self.og - self.hat_o_next
        self.hat_c = tf.einsum('bst,tp,bsp->bs', self.d_to_goal, self.D, self.d_to_goal)

        with tf.name_scope("train"):
            # sum of squared errors in latent space at each time step
            state_prediction_error = tf.reduce_sum(tf.pow(self.hat_o - self.hat_o_next, 2), axis=2)
            self.state_prediction_loss = tf.reduce_mean(state_prediction_error, name='state_prediction_loss')
            self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c,
                                                                     scope='cost_prediction_loss')
            self.flat_weights = tf.concat(
                (tf.reshape(self.A, [-1]), tf.reshape(self.B, [-1]), tf.reshape(self.C, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(self.flat_weights) * self.beta

            self.loss = tf.add_n([self.state_prediction_loss, self.cost_prediction_loss, self.regularization])

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

            self.finish_setup()

    def metadata(self):
        metadata = {
            'tf_version': str(tf.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'beta': self.beta,
            'commandline': self.args_dict['commandline'],
            'dt': self.dt,
            'n_steps': self.n_steps,
        }
        return metadata

    def build_feed_dict(self, x, y, **kwargs):
        s = x['states']
        u = x['actions']
        goal = kwargs['goal']
        c = self.compute_cost_label(s, goal)
        feed_dict = {self.s: s,
                     self.u: u,
                     self.g: goal,
                     self.c: c}
        return feed_dict

    def evaluate(self, eval_x, goal, display=True):
        s = eval_x['states']
        u = eval_x['actions']
        c = self.compute_cost_label(s, goal)
        feed_dict = {self.s: s,
                     self.u: u,
                     self.g: goal,
                     self.c: c}
        ops = [self.A, self.B, self.C, self.D, self.cost_prediction_loss, self.state_prediction_loss,
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
            controllable = self.is_controllable()
            if controllable:
                controllable_string = Fore.GREEN + "True" + Fore.RESET
            else:
                controllable_string = Fore.RED + "False" + Fore.RESET
            print("Controllable?: " + controllable_string)

            # visualize a few sample predictions from the testing data
            self.sess.run([self.hat_o_next], feed_dict=feed_dict)

        return A, B, C, D, c_loss, sp_loss, reg, loss

    @staticmethod
    def compute_cost_label(s, goal):
        """ x is 3d.
            first axis is the trajectory.
            second axis is the time step
            third axis is the state data
        """
        # NOTE: Here we compute the label for cost/reward and constraints
        c = np.sum((s[:, :, [0, 1]] - goal[0, [0, 1]]) ** 2, axis=2)
        return c

    def reduce(self, s):
        ss = np.ndarray((self.batch_size, self.n_steps + 1, self.N))
        ss[0, 0] = s
        feed_dict = {self.s: ss}
        ops = [self.hat_o]
        hat_o = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o[0, 0].reshape(self.M, 1)

    def predict(self, o, u):
        """
        :param o: 1xM or Mx1
        :param u: batch_size x n_steps x L
        """
        hat_o = np.ndarray((self.batch_size, self.n_steps + 1, self.M))
        hat_o[0, 0] = np.squeeze(o)
        feed_dict = {self.hat_o: hat_o, self.u: u}
        ops = [self.hat_o_next]
        hat_o_next = self.sess.run(ops, feed_dict=feed_dict)[0]

    def simple_predict(self, o, u):
        A, B, C, D = self.get_ABCD()
        o_next = o + self.dt * np.dot(B, o) + self.dt * np.dot(C, u)
        return o_next

    def predict_cost(self, o, u, g):
        hat_o = np.ndarray((self.batch_size, self.n_steps + 1, self.M))
        hat_o[0, 0] = np.squeeze(o)
        feed_dict = {self.hat_o: hat_o, self.u: u, self.g: g}
        ops = [self.hat_c]
        hat_c_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_c_next

    def predict_from_s(self, s, u):
        feed_dict = {self.s: s, self.u: u}
        ops = [self.hat_o_next]
        hat_o_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_next

    def cost_of_s(self, s, g):
        return self.cost(self.reduce(s), g)

    def cost(self, o, g):
        hat_o_next = np.zeros((self.batch_size, self.n_steps + 1, self.M))
        for i in range(self.batch_size):
            for j in range(self.n_steps + 1):
                hat_o_next[i, j] = np.squeeze(o)
        feed_dict = {self.hat_o_next: hat_o_next, self.g: g}
        ops = [self.hat_c]
        hat_c = self.sess.run(ops, feed_dict=feed_dict)[0]
        hat_c = np.expand_dims(hat_c, axis=0)
        return hat_c

    def is_controllable(self):
        feed_dict = {}
        ops = [self.B, self.C]
        # Normal people use A and B here but I picked stupid variable names
        state_matrix, control_matrix = self.sess.run(ops, feed_dict=feed_dict)
        controllability_matrix = control.ctrb(state_matrix, control_matrix)
        rank = np.linalg.matrix_rank(controllability_matrix)
        return rank == self.M

    def get_dynamics_matrices(self):
        feed_dict = {}
        ops = [self.B, self.C]
        return self.sess.run(ops, feed_dict=feed_dict)

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