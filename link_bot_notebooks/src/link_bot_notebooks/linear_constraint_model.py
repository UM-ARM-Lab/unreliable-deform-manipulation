#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import os
import json

import numpy as np
import control
import tensorflow as tf
from colorama import Fore
from link_bot_notebooks import base_model
from link_bot_notebooks import toy_problem_optimization_common as tpo
from tensorflow.python import debug as tf_debug


def make_constraint_mask(arr, axis=1):
    """ takes in a 2d array and returns two lists of indeces all elements before the first constriant violation """
    arr = arr.squeeze()
    invalid_val = arr.shape[1]
    mask = arr != 0
    indeces_of_first_violation = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    batch_indeces = []
    time_indeces = []
    for batch_index, index_of_first_violation in enumerate(indeces_of_first_violation):
        for index_with_no_violation in range(index_of_first_violation):
            batch_indeces.append(batch_index)
            time_indeces.append(index_with_no_violation)

    return np.vstack((batch_indeces, time_indeces)).T


@tf.custom_gradient
def sdf_func(sdf, full_sdf_gradient, resolution, sdf_origin_coordinate, sdf_coordinates, P, Q):
    integer_coordinates = tf.cast(tf.divide(sdf_coordinates, resolution), dtype=tf.int32)
    integer_coordinates = tf.reshape(integer_coordinates, [-1, P])
    integer_coordinates = integer_coordinates + sdf_origin_coordinate
    # blindly assume the point is within our grid

    # https://github.com/tensorflow/tensorflow/pull/15857
    # "on CPU an error will be returned and on GPU 0 value will be filled to the expected positions of the output."
    # TODO: make this handle out of bounds correctly. I think correctly for us means return large number for SDF
    # and a gradient towards the origin
    sdf_value = tf.gather_nd(sdf, integer_coordinates, name='index_sdf')
    sdf_value = tf.reshape(sdf_value, (sdf_coordinates.shape[0], sdf_coordinates.shape[1], Q), name='sdfs')

    def __sdf_gradient_func(dy):
        sdf_gradient = tf.gather_nd(full_sdf_gradient, integer_coordinates, name='index_sdf_gradient')
        sdf_gradient = tf.reshape(sdf_gradient, (sdf_coordinates.shape[0], sdf_coordinates.shape[1], P))
        return None, None, None, None, dy * sdf_gradient, None, None

    return sdf_value, __sdf_gradient_func


class LinearConstraintModel(base_model.BaseModel):

    def __init__(self, args, numpy_sdf, numpy_sdf_gradient, numpy_sdf_resolution, batch_size, N, M, L, P, Q, dt,
                 n_steps):
        base_model.BaseModel.__init__(self, N, M, L, P)

        self.seed = args['seed']
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)

        self.batch_size = batch_size
        self.args = args
        self.N = N
        self.M = M
        self.L = L
        self.P = P
        self.Q = Q
        self.beta = 1e-8
        self.n_steps = n_steps
        self.dt = dt
        self.sdf_rows, self.sdf_cols = numpy_sdf.shape
        self.sdf_origin_coordinate = np.array([self.sdf_rows / 2, self.sdf_cols / 2], dtype=np.int32)

        self.s = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1, N), name="s")
        self.u = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps, L), name="u")
        self.s_goal = tf.placeholder(tf.float32, shape=(1, N), name="s_goal")
        self.c_label = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1), name="c")
        self.k_label = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1, Q), name="k")
        self.k_mask_indeces_2d = tf.placeholder(tf.int32, shape=(None, 2), name="k_mask_indeces_2d")
        self.k_label_int = tf.cast(self.k_label, tf.int32)

        # R_d_init = np.random.randn(N, M).astype(np.float32) * 1e-1
        # A_d_init = np.random.randn(M, M).astype(np.float32) * 1e-3
        # B_d_init = np.random.randn(M, L).astype(np.float32) * 1e-1
        # R_k_init = np.random.randn(N, P).astype(np.float32) * 1e-1
        # A_k_init = np.random.randn(P, P).astype(np.float32) * 1e-3
        # B_k_init = np.random.randn(P, L).astype(np.float32) * 1e-1
        # k_threshold_init = np.random.rand() * 1e0

        # IDEAL INIT
        R_d_init = np.zeros((N, M), dtype=np.float32)
        R_d_init[0, 0] = 1
        R_d_init[1, 1] = 1
        A_d_init = np.zeros((M, M), dtype=np.float32)
        B_d_init = np.zeros((M, L), dtype=np.float32)
        np.fill_diagonal(B_d_init, 1.0)
        R_k_init = np.zeros((N, P), dtype=np.float32)
        R_k_init[N - 2, 0] = 1.0
        R_k_init[N - 1, 1] = 1.0
        A_k_init = np.zeros((P, P), dtype=np.float32)
        B_k_init = np.zeros((P, L), dtype=np.float32)
        np.fill_diagonal(B_k_init, 1)
        k_threshold_init = 0.20

        self.R_d = tf.get_variable("R_d", initializer=R_d_init)
        self.A_d = tf.get_variable("A_d", initializer=A_d_init)
        self.B_d = tf.get_variable("B_d", initializer=B_d_init)

        self.R_k = tf.get_variable("R_k", initializer=R_k_init)
        self.A_k = tf.get_variable("A_k", initializer=A_k_init)
        self.B_k = tf.get_variable("B_k", initializer=B_k_init)

        self.threshold_k = tf.get_variable("threshold_k", initializer=k_threshold_init)

        # we force D to be identity because it's tricky to constrain it to be positive semi-definite
        self.D = tf.get_variable("D", initializer=np.eye(self.M, dtype=np.float32), trainable=False)

        self.hat_o_d = tf.einsum('bsn,nm->bsm', self.s, self.R_d, name='hat_o_d')
        self.hat_o_k = tf.einsum('bsn,nm->bsm', self.s, self.R_k, name='hat_o_k')
        self.o_d_goal = tf.matmul(self.s_goal, self.R_d, name='og')

        hat_o_d_next = [self.hat_o_d[:, 0, :]]
        hat_o_k_next = [self.hat_o_k[:, 0, :]]

        for i in range(1, self.n_steps + 1):
            Adod = tf.einsum('mp,bp->bm', self.dt * self.A_d, hat_o_d_next[i - 1], name='A_d_o_d')
            Bdu = tf.einsum('ml,bl->bm', self.dt * self.B_d, self.u[:, i - 1], name='B_d_u')
            hat_o_d_next.append(hat_o_d_next[i - 1] + Adod + Bdu)
            Akok = tf.einsum('mp,bp->bm', self.dt * self.A_k, hat_o_k_next[i - 1], name='A_k_o_k')
            Bku = tf.einsum('ml,bl->bm', self.dt * self.B_k, self.u[:, i - 1], name='B_k_u')
            hat_o_k_next.append(hat_o_k_next[i - 1] + Akok + Bku)

        self.hat_o_d_next = tf.transpose(tf.stack(hat_o_d_next), [1, 0, 2], name='hat_o_d_next')
        self.hat_o_k_next = tf.transpose(tf.stack(hat_o_k_next), [1, 0, 2], name='hat_o_k_next')

        # not using predictions in latent space, just projects of the true state to the latent space
        self.d_to_goal = self.o_d_goal - self.hat_o_d
        self.hat_c = tf.einsum('bst,tp,bsp->bs', self.d_to_goal, self.D, self.d_to_goal, name='hat_c')
        self.sdfs = sdf_func(numpy_sdf, numpy_sdf_gradient, numpy_sdf_resolution, self.sdf_origin_coordinate,
                             self.hat_o_k, self.P, self.Q)
        # because the sigmoid is not very sharp (since meters are very large), this doesn't give a very sharp boundary for
        # the decision between in collision or not. The trade of is this might cause vanishing gradients
        self.hat_k = 100 * (self.threshold_k - self.sdfs)
        self.hat_k_violated = tf.cast(self.sdfs < self.threshold_k, dtype=tf.int32, name="hat_k_violated")
        _, self.constraint_prediction_accuracy = tf.metrics.accuracy(labels=self.k_label,
                                                                     predictions=self.hat_k_violated)

        # NOTE: we use a mask to set the state prediction loss to 0 when the constraint is violated?
        # this way we don't penalize our model for failing to predict the dynamics in collision
        # the (1-label) inverts the labels so that 0 means in collision and mask out
        # TODO: this is invalid for when Q is not 1

        with tf.name_scope("train"):
            # sum of squared errors in latent space at each time step
            with tf.name_scope("latent_dynamics_d"):
                self.all_state_prediction_error_in_d = tf.reduce_sum(tf.pow(self.hat_o_d - self.hat_o_d_next, 2),
                                                                     axis=2)
                self.state_prediction_error_in_d = tf.gather_nd(self.all_state_prediction_error_in_d,
                                                                self.k_mask_indeces_2d,
                                                                name='all_state_prediction_error_in_d')
                self.state_prediction_loss_in_d = tf.reduce_mean(self.state_prediction_error_in_d,
                                                                 name='state_prediction_loss_in_d')
                self.all_cost_prediction_error = tf.square(self.hat_c - self.c_label, name='all_cost_prediction_error')
                self.cost_prediction_error = tf.gather_nd(self.all_cost_prediction_error, self.k_mask_indeces_2d,
                                                          name='cost_prediction_error')
                self.cost_prediction_loss = tf.reduce_mean(self.cost_prediction_error, name='cost_prediction_loss')

            with tf.name_scope("latent_constraints_k"):
                self.all_state_prediction_error_in_k = tf.reduce_sum(tf.pow(self.hat_o_k - self.hat_o_k_next, 2),
                                                                     axis=2)
                self.state_prediction_error_in_k = tf.gather_nd(self.all_state_prediction_error_in_k,
                                                                self.k_mask_indeces_2d,
                                                                name='all_state_prediction_error_in_k')
                self.top_state_prediction_error_in_k, self.top_state_prediction_error_in_k_indeces \
                    = tf.math.top_k(self.state_prediction_error_in_k, k=100)
                self.state_prediction_loss_in_k = tf.reduce_mean(self.state_prediction_error_in_k,
                                                                 name='state_prediction_loss_in_k')

                self.constraint_prediction_error = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.hat_k,
                                                                                           labels=self.k_label,
                                                                                           name='constraint_prediction_error')
                self.constraint_prediction_loss = tf.reduce_mean(self.constraint_prediction_error,
                                                                 name="constraint_prediction_loss")

            self.flat_weights = tf.concat(
                (tf.reshape(self.R_d, [-1]), tf.reshape(self.A_d, [-1]), tf.reshape(self.B_d, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(self.flat_weights) * self.beta

            self.loss = tf.add_n([self.state_prediction_loss_in_d,
                                  self.cost_prediction_loss,
                                  self.state_prediction_loss_in_k,
                                  self.constraint_prediction_loss,
                                  ], name='loss')

            self.global_step = tf.get_variable("global_step", initializer=0, trainable=False)
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, global_step=self.global_step)

            trainable_vars = tf.trainable_variables()
            for var in trainable_vars:
                name = var.name.replace(":", "_")
                grads = tf.gradients(self.loss, var, name='dLoss_d{}'.format(name))
                for grad in grads:
                    if grad is not None:
                        tf.summary.histogram(name + "/gradient", grad)
                    else:
                        print("Warning... there is no gradient of the loss with respect to {}".format(var.name))

            tf.summary.scalar("constraint_prediction_accuracy_summary", self.constraint_prediction_accuracy)
            tf.summary.scalar("k_threshold_summary", self.threshold_k)
            tf.summary.scalar("constraint_prediction_loss_summary", self.constraint_prediction_loss)
            tf.summary.scalar("state_prediction_loss_in_d_summary", self.state_prediction_loss_in_d)
            tf.summary.scalar("state_prediction_loss_in_k_summary", self.state_prediction_loss_in_k)
            tf.summary.scalar("cost_prediction_loss_summary", self.cost_prediction_loss)
            tf.summary.scalar("regularization_loss_summary", self.regularization)
            tf.summary.scalar("loss_summary", self.loss)

            self.summaries = tf.summary.merge_all()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
                'seed': self.args['seed'],
                'checkpoint': self.args['checkpoint'],
                'N': self.N,
                'M': self.M,
                'L': self.L,
                'P': self.P,
                'Q': self.Q,
                'beta': self.beta,
                'dt': self.dt,
                'commandline': self.args['commandline'],
            }
            metadata_file.write(json.dumps(metadata, indent=2))

            writer = tf.summary.FileWriter(full_log_path)
            writer.add_graph(self.sess.graph)

        try:
            s = train_x['states']
            u = train_x['actions']
            c = self.compute_cost_label(s, goal)
            if 'constraints' in train_x:
                k = train_x['constraints']
            else:
                print("WARNING: no constraint data given")
                k = np.zeros((s.shape[0], s.shape[1], self.Q))
            mask = make_constraint_mask(k)
            feed_dict = {self.s: s,
                         self.u: u,
                         self.s_goal: goal,
                         self.c_label: c,
                         self.k_label: k,
                         self.k_mask_indeces_2d: mask}

            ops = [self.global_step, self.summaries, self.loss, self.opt]
            for i in range(epochs):
                step, summary, loss, _ = self.sess.run(ops, feed_dict=feed_dict)

                if 'save_period' in self.args and (step % self.args['save_period'] == 0 or step == 1):
                    if self.args['log'] is not None:
                        writer.add_summary(summary, step)
                        self.save(full_log_path, loss=loss)

                if 'print_period' in self.args and (step % self.args['print_period'] == 0 or step == 1):
                    print('step: {}, loss: {} '.format(step, loss))

        except KeyboardInterrupt:
            print("stop!!!")
            interrupted = True
            pass
        finally:
            ops = [self.R_d, self.A_d, self.B_d, self.D]
            A, B, C, D = self.sess.run(ops, feed_dict={})
            if self.args['verbose']:
                print("Loss: {}".format(loss))
                print("A:\n{}".format(A))
                print("B:\n{}".format(B))
                print("C:\n{}".format(C))
                print("D:\n{}".format(D))

        return interrupted

    def evaluate(self, eval_x, goal, display=True):
        s = eval_x['states']
        u = eval_x['actions']
        c = self.compute_cost_label(s, goal)
        if 'constraints' in eval_x:
            k = eval_x['constraints']
        else:
            k = np.zeros((s.shape[0], s.shape[1], self.Q))
        feed_dict = {self.s: s,
                     self.u: u,
                     self.s_goal: goal,
                     self.c_label: c,
                     self.k_label: k,
                     self.k_mask_indeces_2d: make_constraint_mask(k)}
        ops = [self.R_d, self.A_d, self.B_d, self.D, self.R_k, self.A_k, self.B_k, self.threshold_k,
               self.state_prediction_loss_in_d, self.state_prediction_loss_in_k, self.cost_prediction_loss,
               self.constraint_prediction_loss, self.regularization, self.loss, self.constraint_prediction_accuracy]
        R_d, A_d, B_d, D, R_k, A_k, B_k, threshold_k, spd_loss, spk_loss, c_loss, k_loss, reg, loss, k_accuracy = \
            self.sess.run(ops, feed_dict=feed_dict)

        if display:
            print("State Prediction Loss in d: {}".format(spd_loss))
            print("State Prediction Loss in k: {}".format(spk_loss))
            print("Cost Loss: {}".format(c_loss))
            print("Constraint Loss: {}".format(k_loss))
            print("Regularization: {}".format(reg))
            print("Overall Loss: {}".format(loss))
            print("R_d:\n{}".format(R_d))
            print("A_d:\n{}".format(A_d))
            print("B_d:\n{}".format(B_d))
            print("D:\n{}".format(D))
            print("R_k:\n{}".format(R_k))
            print("A_k:\n{}".format(A_k))
            print("B_k:\n{}".format(B_k))
            print("threshold_k:\n{}".format(threshold_k))
            print("constraint prediction accuracy:\n{}".format(k_accuracy))
            controllable = self.is_controllable()
            if controllable:
                controllable_string = Fore.GREEN + "True" + Fore.RESET
            else:
                controllable_string = Fore.RED + "False" + Fore.RESET
            print("Controllable?: " + controllable_string)

            # visualize a few sample predictions from the testing data
            self.sess.run([self.hat_o_d_next], feed_dict=feed_dict)

        return R_d, A_d, B_d, D, R_k, A_k, B_k, c_loss, spd_loss, spk_loss, c_loss, k_loss

    @staticmethod
    def compute_cost_label(s, goal):
        """ x is 3d.
            first axis is the trajectory.
            second axis is the time step
            third axis is the state data
        """
        c = np.sum((s[:, :, [0, 1]] - goal[0, [0, 1]]) ** 2, axis=2)
        return c

    def setup(self):
        if self.args['checkpoint']:
            self.sess.run([tf.local_variables_initializer()])
            self.load()
        else:
            self.init()

    def init(self):
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def reduce(self, s):
        # make dummy size array and just fill in
        ss = np.ndarray((self.batch_size, self.n_steps + 1, self.N))
        ss[0, 0] = s
        feed_dict = {self.s: ss}
        ops = [self.hat_o_d, self.hat_o_k]
        hat_o_d, hat_o_k = self.sess.run(ops, feed_dict=feed_dict)
        return hat_o_d[0, 0].reshape(self.M, 1), hat_o_k[0, 0].reshape(self.P, 1)

    def predict(self, o, u):
        """
        :param o: 1xM or Mx1
        :param u: batch_size x n_steps x L
        """
        hat_o = np.ndarray((self.batch_size, self.n_steps + 1, self.M))
        hat_o[0, 0] = np.squeeze(o)
        feed_dict = {self.hat_o_d: hat_o, self.u: u}
        ops = [self.hat_o_d_next]
        hat_o_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_next

    def simple_dual_predict(self, o_d, o_k, u):
        A_d, B_d, A_k, B_k = self.get_dynamics_matrices()
        o_d_next = o_d + self.dt * np.dot(A_d, o_d) + self.dt * np.dot(B_d, u)
        o_k_next = o_k + self.dt * np.dot(A_k, o_k) + self.dt * np.dot(B_k, u)
        return o_d_next, o_k_next

    def simple_predict_constraint(self, o_k, u):
        _, _, A_k, B_k = self.get_dynamics_matrices()
        o_k_next = o_k + self.dt * np.dot(A_k, o_k) + self.dt * np.dot(B_k, u)
        return o_k_next

    def simple_predict(self, o_d, u):
        A_d, B_d, _, _ = self.get_dynamics_matrices()
        o_d_next = o_d + self.dt * np.dot(A_d, o_d) + self.dt * np.dot(B_d, u)
        return o_d_next

    def predict_cost(self, o, u, g):
        hat_o = np.ndarray((self.batch_size, self.n_steps + 1, self.M))
        hat_o[0, 0] = np.squeeze(o)
        feed_dict = {self.hat_o_d: hat_o, self.u: u, self.s_goal: g}
        ops = [self.hat_c]
        hat_c_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_c_next

    def constraint_violated(self, o_k):
        full_o_k = np.ndarray((1, self.n_steps + 1, self.P))
        full_o_k[0, 0] = o_k
        feed_dict = {self.hat_o_k: full_o_k}
        ops = [self.hat_k_violated]
        constraint_violated = self.sess.run(ops, feed_dict=feed_dict)
        # take the first op from the list, then take the first batch and first time step from that
        constraint_violated = constraint_violated[0][0, 0]
        return constraint_violated

    def predict_from_s(self, s, u):
        feed_dict = {self.s: s, self.u: u}
        ops = [self.hat_o_d_next]
        hat_o_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_next

    def cost_of_s(self, s, g):
        return self.cost(self.reduce(s), g)

    def cost(self, o, g):
        hat_o_next = np.zeros((self.batch_size, self.n_steps + 1, self.M))
        for i in range(self.batch_size):
            for j in range(self.n_steps + 1):
                hat_o_next[i, j] = np.squeeze(o)
        feed_dict = {self.hat_o_d_next: hat_o_next, self.s_goal: g}
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

    def is_controllable(self):
        feed_dict = {}
        ops = [self.A_d, self.B_d]
        # Normal people use A and B here but I picked stupid variable names
        state_matrix, control_matrix = self.sess.run(ops, feed_dict=feed_dict)
        controllability_matrix = control.ctrb(state_matrix, control_matrix)
        rank = np.linalg.matrix_rank(controllability_matrix)
        return rank == self.M

    def get_matrices(self):
        feed_dict = {}
        ops = [self.R_d, self.A_d, self.B_d, self.D, self.R_k, self.A_k, self.B_k]
        return self.sess.run(ops, feed_dict=feed_dict)

    def get_dynamics_matrices(self):
        feed_dict = {}
        ops = [self.A_d, self.B_d, self.A_k, self.B_k]
        return self.sess.run(ops, feed_dict=feed_dict)

    def get_R_d(self):
        feed_dict = {}
        ops = [self.R_d]
        R_d = self.sess.run(ops, feed_dict=feed_dict)[0]
        return R_d

    def __str__(self):
        ops = [self.R_d, self.A_d, self.B_d, self.R_k, self.A_k, self.B_k, self.threshold_k]
        return "R_d:\n{}\nA_d:\n{}\nB_d:\n{}\nR_k:\n{}\nA_k:\n{}\nB_k:\n{}\n\nthreshold_k:\n{}\n".format(
            *self.sess.run(ops, feed_dict={}))
