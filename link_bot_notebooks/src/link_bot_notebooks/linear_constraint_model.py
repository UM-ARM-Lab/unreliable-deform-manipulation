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

# TODO: make this a proper input not a global variable
# assume environment is bounded (-12, 12) in x and y which means we need 24x24
ROWS = COLS = 24
numpy_sdf = np.zeros((24, 24))
numpy_sdf_gradient = np.zeros((24, 24, 2))


def point_to_index(xy):
    pt_float = xy - np.array([[ROWS / 2], [COLS / 2]])
    return pt_float.astype(np.int32)


# hard code the SDF value of anything within [(1,1),(1,2),(2,2),(2,1)] as SDF value = 1
numpy_sdf[point_to_index(np.array([2, 0]))] = 1
numpy_sdf_gradient[point_to_index(np.array([2, 0]))] = [0, 0]
numpy_sdf_gradient[point_to_index(np.array([3, 0]))] = [1, 0]
numpy_sdf_gradient[point_to_index(np.array([2, 1]))] = [0, 1]
numpy_sdf_gradient[point_to_index(np.array([1, 0]))] = [-1, 0]
numpy_sdf_gradient[point_to_index(np.array([2, -1]))] = [0, -1]
numpy_sdf_gradient[point_to_index(np.array([3, 1]))] = [1 / np.sqrt(2), 1 / np.sqrt(2)]
numpy_sdf_gradient[point_to_index(np.array([3, -1]))] = [1 / np.sqrt(2), -1 / np.sqrt(2)]
numpy_sdf_gradient[point_to_index(np.array([1, 1]))] = [-1 / np.sqrt(2), 1 / np.sqrt(2)]
numpy_sdf_gradient[point_to_index(np.array([1, -1]))] = [-1 / np.sqrt(2), -1 / np.sqrt(2)]
resolution = np.array([1, 1], dtype=np.float32)


@tf.custom_gradient
def sdf(placeholder_sdf, placeholder_sdf_gradient, point, P, Q):
    integer_coordinates = tf.cast(tf.divide(point, resolution), dtype=tf.int32)
    integer_coordinates = tf.reshape(integer_coordinates, [-1, P])
    # blindly assume the point is within our grid
    sdf_value = tf.gather_nd(placeholder_sdf, integer_coordinates, name='index_sdf')
    sdf_value = tf.reshape(sdf_value, (point.shape[0], point.shape[1], Q))

    # noinspection PyUnusedLocal
    def __sdf_gradient_func(dy):
        sdf_gradient = tf.gather_nd(placeholder_sdf_gradient, integer_coordinates, name='index_sdf_gradient')
        sdf_gradient = tf.reshape(sdf_gradient, (point.shape[0], point.shape[1], P))
        return None, None, sdf_gradient, None, None

    return sdf_value, __sdf_gradient_func


class LinearConstraintModel(base_model.BaseModel):

    def __init__(self, args, batch_size, N, M, L, P, Q, dt, n_steps, seed=0):
        base_model.BaseModel.__init__(self, N, M, L, P)

        self.seed = seed
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

        self.s = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1, N), name="s")
        self.u = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps, L), name="u")
        self.s_goal = tf.placeholder(tf.float32, shape=(1, N), name="s_goal")
        self.c_label = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1), name="c")
        self.k_label = tf.placeholder(tf.float32, shape=(batch_size, self.n_steps + 1, Q), name="k")
        self.placeholder_sdf = tf.placeholder(tf.float32, shape=(ROWS, COLS), name="sdf")
        self.placeholder_sdf_gradient = tf.placeholder(tf.float32, shape=(ROWS, COLS, 2), name="sdf_gradient")

        R_d_init = np.random.randn(N, M).astype(np.float32) * 1e-6
        R_d_init[0, 0] = 1
        R_d_init[1, 1] = 1
        R_k_init = np.random.randn(N, P).astype(np.float32) * 1e-6
        R_k_init[0, 0] = 1
        R_k_init[1, 1] = 1
        B_init = np.random.randn(M, M).astype(np.float32) * 1e-6
        C_init = np.random.randn(M, L).astype(np.float32) * 1e-6
        np.fill_diagonal(C_init, 1)

        self.R_d = tf.get_variable("R_d", initializer=R_d_init)
        self.B_d = tf.get_variable("B_d", initializer=B_init)
        self.C_d = tf.get_variable("C_d", initializer=C_init)

        self.R_k = tf.get_variable("R_k", initializer=R_k_init)
        self.B_k = tf.get_variable("B_k", initializer=B_init)
        self.C_k = tf.get_variable("C_k", initializer=C_init)

        # we force D to be identity because it's tricky to constrain it to be positive semi-definite
        self.D = tf.get_variable("D", initializer=np.eye(self.M, dtype=np.float32), trainable=False)

        self.hat_o_d = tf.einsum('bsn,nm->bsm', self.s, self.R_d, name='hat_o_d')
        self.hat_o_k = tf.einsum('bsn,nm->bsm', self.s, self.R_k, name='hat_o_k')
        self.o_d_goal = tf.matmul(self.s_goal, self.R_d, name='og')

        hat_o_d_next = [self.hat_o_d[:, 0, :]]
        hat_o_k_next = [self.hat_o_k[:, 0, :]]

        for i in range(1, self.n_steps + 1):
            Bdod = tf.einsum('mp,bp->bm', self.dt * self.B_d, hat_o_d_next[i - 1], name='B_d_o_d')
            Cdu = tf.einsum('ml,bl->bm', self.dt * self.C_d, self.u[:, i - 1], name='C_d_u')
            hat_o_d_next.append(hat_o_d_next[i - 1] + Bdod + Cdu)
            Bkok = tf.einsum('mp,bp->bm', self.dt * self.B_k, hat_o_k_next[i - 1], name='B_k_o_k')
            Cku = tf.einsum('ml,bl->bm', self.dt * self.C_k, self.u[:, i - 1], name='C_k_u')
            hat_o_k_next.append(hat_o_k_next[i - 1] + Bkok + Cku)

        self.hat_o_d_next = tf.transpose(tf.stack(hat_o_d_next), [1, 0, 2], name='hat_o_d_next')
        self.hat_o_k_next = tf.transpose(tf.stack(hat_o_k_next), [1, 0, 2], name='hat_o_k_next')

        self.d_to_goal = self.o_d_goal - self.hat_o_d_next
        self.hat_c = tf.einsum('bst,tp,bsp->bs', self.d_to_goal, self.D, self.d_to_goal)
        self.hat_k = sdf(self.placeholder_sdf, self.placeholder_sdf_gradient, self.hat_o_k_next, self.P, self.Q)

        with tf.name_scope("train"):
            # sum of squared errors in latent space at each time step
            with tf.name_scope("latent_dynamics_o"):
                state_prediction_error_in_o = tf.reduce_sum(tf.pow(self.hat_o_d - self.hat_o_d_next, 2), axis=2)
                self.state_prediction_loss_in_o = tf.reduce_mean(state_prediction_error_in_o,
                                                                 name='state_prediction_loss_in_o')
                self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c_label,
                                                                         predictions=self.hat_c,
                                                                         scope='cost_prediction_loss')
            with tf.name_scope("latent_constraints_k"):
                state_prediction_error_in_k = tf.reduce_sum(tf.pow(self.hat_o_k - self.hat_o_k_next, 2), axis=2)
                self.state_prediction_loss_in_k = tf.reduce_mean(state_prediction_error_in_k,
                                                                 name='state_prediction_loss_in_k')
                self.constraint_prediction_loss = tf.losses.mean_squared_error(labels=self.k_label,
                                                                               predictions=self.hat_k,
                                                                               scope='constraint_prediction_loss')

            self.flat_weights = tf.concat(
                (tf.reshape(self.R_d, [-1]), tf.reshape(self.B_d, [-1]), tf.reshape(self.C_d, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(self.flat_weights) * self.beta

            self.loss = tf.add_n(
                [self.state_prediction_loss_in_o, self.state_prediction_loss_in_k, self.cost_prediction_loss,
                 self.constraint_prediction_loss, self.regularization])

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

            tf.summary.scalar("state_prediction_loss_in_o", self.state_prediction_loss_in_o)
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
                         self.s_goal: goal,
                         self.c_label: c,
                         self.placeholder_sdf: numpy_sdf,
                         self.placeholder_sdf_gradient: numpy_sdf_gradient}
            ops = [self.global_step, self.summaries, self.loss, self.opt]
            for i in range(epochs):
                step, summary, loss, _ = self.sess.run(ops, feed_dict=feed_dict)

                if 'save_period' in self.args and (step % self.args['save_period'] == 0 or step == 1):
                    if self.args['log'] is not None:
                        writer.add_summary(summary, step)
                        self.save(full_log_path, loss=loss)

                if 'print_period' in self.args and (step % self.args['print_period'] == 0 or step == 1):
                    print(step, loss)

        except KeyboardInterrupt:
            print("stop!!!")
            interrupted = True
            pass
        finally:
            ops = [self.R_d, self.B_d, self.C_d, self.D]
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
                     self.s_goal: goal,
                     self.c_label: c}
        ops = [self.R_d, self.B_d, self.C_d, self.D, self.cost_prediction_loss, self.state_prediction_loss_in_o,
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
            self.sess.run([self.hat_o_d_next], feed_dict=feed_dict)

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
        ss = np.ndarray((self.batch_size, self.n_steps + 1, self.N))
        ss[0, 0] = s
        feed_dict = {self.s: ss}
        ops = [self.hat_o_d]
        hat_o = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o[0, 0].reshape(self.M, 1)

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

    def simple_predict(self, o, u):
        A, B, C, D = self.get_ABCD()
        o_next = o + self.dt * np.dot(B, o) + self.dt * np.dot(C, u)
        return o_next

    def predict_cost(self, o, u, g):
        hat_o = np.ndarray((self.batch_size, self.n_steps + 1, self.M))
        hat_o[0, 0] = np.squeeze(o)
        feed_dict = {self.hat_o_d: hat_o, self.u: u, self.s_goal: g}
        ops = [self.hat_c]
        hat_c_next = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_c_next

    def hat_constraint(self, s):
        feed_dict = {self.s: s,
                     self.placeholder_sdf: numpy_sdf,
                     self.placeholder_sdf_gradient: numpy_sdf_gradient,
                     }
        ops = [self.hat_o_k]
        hat_constraint = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_constraint

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
        ops = [self.B_d, self.C_d]
        # Normal people use A and B here but I picked stupid variable names
        state_matrix, control_matrix = self.sess.run(ops, feed_dict=feed_dict)
        controllability_matrix = control.ctrb(state_matrix, control_matrix)
        rank = np.linalg.matrix_rank(controllability_matrix)
        return rank == self.M

    def get_ABCD(self):
        feed_dict = {}
        ops = [self.R_d, self.B_d, self.C_d, self.D]
        return self.sess.run(ops, feed_dict=feed_dict)

    def get_A(self):
        feed_dict = {}
        ops = [self.R_d]
        A = self.sess.run(ops, feed_dict=feed_dict)[0]
        return A

    def __str__(self):
        A, B, C, D = self.get_ABCD()
        return "A:\n" + np.array2string(A) + "\n" + \
               "B:\n" + np.array2string(B) + "\n" + \
               "C:\n" + np.array2string(C) + "\n" + \
               "D:\n" + np.array2string(D) + "\n"
