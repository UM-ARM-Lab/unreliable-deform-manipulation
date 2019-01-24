#!/usr/bin/env python
from __future__ import print_function

import os
import json

import numpy as np
import tensorflow as tf
from colorama import Fore
from link_bot_notebooks import base_model
from link_bot_notebooks import toy_problem_optimization_common as tpo

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


@tf.custom_gradient
def sdf(placeholder_sdf, placeholder_sdf_gradient, point):
    resolution = np.array([[1], [1]], dtype=np.float32)
    integer_coordinates = tf.cast(tf.divide(point, resolution), dtype=tf.int32)
    # blindly assume the point is within our grid
    sdf_value = tf.gather_nd(placeholder_sdf, tf.transpose(integer_coordinates), name='index_sdf')

    # noinspection PyUnusedLocal
    def __sdf_gradient_func(dy):
        sdf_gradient = tf.gather_nd(placeholder_sdf_gradient, tf.transpose(integer_coordinates))
        return None, None, tf.transpose(sdf_gradient)

    return sdf_value, __sdf_gradient_func


class LinearInvertibleModel(base_model.BaseModel):

    def __init__(self, args, N, M, K, L, dt, seed=0):
        base_model.BaseModel.__init__(self, N, M, L)

        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        self.args = args
        self.N = N
        self.M_control = M
        self.M_constraint = K
        self.L = L
        self.beta = 1e-8
        self.dt = dt

        self.s = tf.placeholder(tf.float32, shape=(N, None), name="s")
        self.s_ = tf.placeholder(tf.float32, shape=(N, None), name="s_")
        self.u = tf.placeholder(tf.float32, shape=(L, None), name="u")
        self.g = tf.placeholder(tf.float32, shape=(N, 1), name="g")
        self.c = tf.placeholder(tf.float32, shape=(None), name="c")
        self.placeholder_sdf = tf.placeholder(tf.float32, shape=(ROWS, COLS), name="sdf")
        self.placeholder_sdf_gradient = tf.placeholder(tf.float32, shape=(ROWS, COLS, 2), name="sdf_gradient")
        self.constraint = tf.placeholder(tf.float32, shape=(None), name="constraint")
        self.c_ = tf.placeholder(tf.float32, shape=(None), name="c_")

        self.A_control = tf.Variable(tf.truncated_normal(shape=[self.M_control, N]), name="A_control", dtype=tf.float32)
        self.A_constraint = tf.Variable(tf.truncated_normal(shape=[self.M_constraint, self.M_control]),
                                        name="A_constraint", dtype=tf.float32)
        self.B = tf.Variable(tf.truncated_normal(shape=[self.M_control, self.M_control]), name="B", dtype=tf.float32)
        self.C = tf.Variable(tf.truncated_normal(shape=[self.M_control, L]), name="C", dtype=tf.float32)
        self.B_inv = tf.Variable(tf.truncated_normal(shape=[self.M_control, self.M_control]), name="B_inv",
                                 dtype=tf.float32)
        self.C_inv = tf.Variable(tf.truncated_normal(shape=[L, self.M_control]), name="C_inv", dtype=tf.float32)
        self.D = tf.Variable(np.eye(self.M_control, dtype=np.float32), trainable=False)

        self.hat_o_control = tf.matmul(self.A_control, self.s, name='reduce')
        self.hat_o_constraint = tf.matmul(self.A_constraint, self.hat_o_control, name='hat_o_constraint')
        self.og = tf.matmul(self.A_control, self.g, name='reduce_goal')
        self.o_control_ = tf.matmul(self.A_control, self.s_, name='reduce_')

        self.state_bo = tf.matmul(self.dt * self.B, self.hat_o_control, name='dynamics')
        self.state_o_ = self.hat_o_control + self.state_bo
        self.temp_o_control = tf.matmul(self.dt * self.C, self.u, name='controls')
        self.hat_o_control_ = tf.add(self.state_o_, self.temp_o_control, name='hat_o_')

        self.hat_constraint = sdf(self.placeholder_sdf, self.placeholder_sdf_gradient, self.hat_o_constraint)

        self.d_to_goal = self.og - self.hat_o_control
        self.d_to_goal_ = self.og - self.hat_o_control_
        self.hat_c = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal), self.D), self.d_to_goal))
        self.hat_c_ = tf.linalg.tensor_diag_part(
            tf.matmul(tf.matmul(tf.transpose(self.d_to_goal_), self.D), self.d_to_goal_))

        self.hat_u = tf.matmul(self.C_inv, 1.0 / self.dt * (
                self.hat_o_control_ - self.hat_o_control) - tf.matmul(self.B_inv, self.hat_o_control))

        with tf.name_scope("train"):
            self.cost_loss = tf.losses.mean_squared_error(labels=self.c, predictions=self.hat_c)
            self.state_prediction_loss = tf.reduce_mean(tf.norm(self.o_control_ - self.hat_o_control_, axis=0))
            self.cost_prediction_loss = tf.losses.mean_squared_error(labels=self.c_, predictions=self.hat_c_)
            self.inv_loss = tf.losses.mean_squared_error(labels=self.u, predictions=self.hat_u)
            self.constraint_loss = tf.losses.mean_squared_error(labels=self.constraint, predictions=self.hat_constraint)
            flat_weights = tf.concat((tf.reshape(self.A_control, [-1]), tf.reshape(self.B, [-1]),
                                      tf.reshape(self.C, [-1]), tf.reshape(self.D, [-1])), axis=0)
            self.regularization = tf.nn.l2_loss(flat_weights) * self.beta
            self.loss = self.cost_loss + self.state_prediction_loss + self.cost_prediction_loss + self.inv_loss \
                        + self.constraint_loss + self.regularization
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
            gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_config))
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
            }
            metadata_file.write(json.dumps(metadata, indent=2))

            writer = tf.summary.FileWriter(full_log_path)
            writer.add_graph(self.sess.graph)

        try:
            s, s_, u, c, c_, constraint = self.batch(train_x, goal)
            feed_dict = {self.s: s,
                         self.s_: s_,
                         self.u: u,
                         self.g: goal,
                         self.placeholder_sdf: numpy_sdf,
                         self.placeholder_sdf_gradient: numpy_sdf_gradient,
                         self.constraint: constraint,
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

    def get_hat_o_control(self, o):
        feed_dict = {self.hat_o_control: o}
        ops = [self.hat_o_constraint]
        hat_o_constraint = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_o_constraint

    def hat_constraint_from_s(self, s):
        feed_dict = {self.s: s,
                     self.placeholder_sdf: numpy_sdf,
                     self.placeholder_sdf_gradient: numpy_sdf_gradient,
                     }
        ops = [self.hat_constraint]
        hat_constraint = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_constraint

    def get_hat_constraint(self, o):
        feed_dict = {self.hat_o_constraint: o,
                     self.placeholder_sdf: numpy_sdf,
                     self.placeholder_sdf_gradient: numpy_sdf_gradient,
                     }
        ops = [self.hat_constraint]
        hat_constraint = self.sess.run(ops, feed_dict=feed_dict)[0]
        return hat_constraint

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
        s, s_, u, c, c_, constraint = self.batch(eval_x, goal)
        feed_dict = {self.s: s,
                     self.s_: s_,
                     self.u: u,
                     self.g: goal,
                     self.constraint: constraint,
                     self.placeholder_sdf: numpy_sdf,
                     self.placeholder_sdf_gradient: numpy_sdf_gradient,
                     self.c: c,
                     self.c_: c_}
        ops = [self.A_control, self.A_constraint, self.B, self.C, self.D, self.cost_loss, self.state_prediction_loss,
               self.cost_prediction_loss, self.constraint_loss, self.regularization, self.loss]
        A_control, A_constraint, B, C, D, c_loss, sp_loss, cp_loss, constraint_loss, reg, loss = self.sess.run(ops, feed_dict=feed_dict)
        xx = self.sess.run(self.hat_constraint, feed_dict=feed_dict)
        np.set_printoptions(threshold=np.nan)
        print(xx)
        if display:
            print("Cost Loss: {}".format(c_loss))
            print("State Prediction Loss: {}".format(sp_loss))
            print("Cost Prediction Loss: {}".format(cp_loss))
            print("Constraint Prediction Loss: {}".format(constraint_loss))
            print("Regularization: {}".format(reg))
            print("Overall Loss: {}".format(loss))
            print("A_control:\n{}".format(A_control))
            print("A_constraint:\n{}".format(A_constraint))
            print("B:\n{}".format(B))
            print("C:\n{}".format(C))
            print("D:\n{}".format(D))
        return A_control, A_constraint, B, C, D, c_loss, sp_loss, cp_loss, reg, loss

    def batch(self, x, goal):
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
        u = batch_examples[0, self.N:, :]

        # Here we compute the label for cost/reward and constraints
        c = np.sum((s[[0, 1], :] - goal[[0, 1]]) ** 2, axis=0)
        c_ = np.sum((s_[[0, 1], :] - goal[[0, 1]]) ** 2, axis=0)
        pts = point_to_index(s[[4, 5], :]).astype(np.int32)
        xs = pts[0, :]
        ys = pts[1, :]
        constraint = numpy_sdf[xs, ys]
        return s, s_, u, c, c_, constraint

    def get_AABCD(self):
        feed_dict = {}
        ops = [self.A_control, self.A_constraint, self.B, self.C, self.D]
        return self.sess.run(ops, feed_dict=feed_dict)

    def get_A(self):
        feed_dict = {}
        ops = [self.A_control]
        A = self.sess.run(ops, feed_dict=feed_dict)[0]
        return A

    def __str__(self):
        if hasattr(self, "sess"):
            A_control, A_constraint, B, C, D = self.get_AABCD()
            return "A_control:\n" + np.array2string(A_control) + "\n" + \
                   "A_constraint:\n" + np.array2string(A_constraint) + "\n" + \
                   "B:\n" + np.array2string(B) + "\n" + \
                   "C:\n" + np.array2string(C) + "\n" + \
                   "D:\n" + np.array2string(D) + "\n"
        else:
            return "Uninitialized_Linear_Invertible_Constraint_Model"
