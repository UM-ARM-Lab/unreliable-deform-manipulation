#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import json
import os

import numpy as np
import tensorflow as tf
from colorama import Fore
from tensorflow.python import debug as tf_debug

import link_bot_notebooks.experiments_util
from link_bot_notebooks import base_model


@tf.custom_gradient
def sdf_func(sdf, full_sdf_gradient, resolution, sdf_origin_coordinate, sdf_coordinates, P):
    integer_coordinates = tf.cast(tf.divide(sdf_coordinates, resolution), dtype=tf.int32)
    integer_coordinates = tf.reshape(integer_coordinates, [-1, P])
    integer_coordinates = integer_coordinates + sdf_origin_coordinate
    # blindly assume the point is within our grid

    # https://github.com/tensorflow/tensorflow/pull/15857
    # "on CPU an error will be returned and on GPU 0 value will be filled to the expected positions of the output."
    # TODO: make this handle out of bounds correctly. I think correctly for us means return large number for SDF
    # and a gradient towards the origin
    sdf_value = tf.gather_nd(sdf, integer_coordinates, name='index_sdf')
    sdf_value = tf.reshape(sdf_value, [-1, 1], name='sdfs')

    def __sdf_gradient_func(dy):
        sdf_gradient = tf.gather_nd(full_sdf_gradient, integer_coordinates, name='index_sdf_gradient')
        sdf_gradient = tf.reshape(sdf_gradient, [-1, P])
        return None, None, None, None, dy * sdf_gradient, None

    return sdf_value, __sdf_gradient_func


class ConstraintModel(base_model.BaseModel):

    def __init__(self, args, numpy_sdf, numpy_sdf_gradient, numpy_sdf_resolution, N):
        base_model.BaseModel.__init__(self, N)

        self.seed = args['seed']
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)

        self.args = args
        self.N = N
        self.beta = 1e-8
        self.sdf_rows, self.sdf_cols = numpy_sdf.shape
        sdf_origin_coordinate = np.array([self.sdf_rows / 2, self.sdf_cols / 2], dtype=np.int32)

        self.observations = tf.placeholder(tf.float32, shape=(None, N), name="observations")
        self.k_label = tf.placeholder(tf.float32, shape=(None, 1), name="k")
        self.k_label_int = tf.cast(self.k_label, tf.int32)

        if args['random_init']:
            # RANDOM INIT
            R_k_init = np.random.randn(N, 1).astype(np.float32) * 1e-1
            k_threshold_init = np.random.rand() * 1e-1
        else:
            # IDEAL INIT
            R_k_init = np.zeros((N, 2), dtype=np.float32)
            R_k_init[N - 2, 0] = 1.0
            R_k_init[N - 1, 1] = 1.0
            k_threshold_init = 0.20

        self.R_k = tf.get_variable("R_k", initializer=R_k_init)

        self.threshold_k = tf.get_variable("threshold_k", initializer=k_threshold_init)

        self.hat_latent_k = tf.einsum('bn,nm->bm', self.observations, self.R_k, name='hat_latent_k')

        self.sdfs = sdf_func(numpy_sdf, numpy_sdf_gradient, numpy_sdf_resolution, sdf_origin_coordinate,
                             self.hat_latent_k, 2)
        # because the sigmoid is not very sharp (since meters are very large),
        # this doesn't give a very sharp boundary for the decision between in collision or not.
        # The trade of is this might cause vanishing gradients
        self.hat_k = 100 * (self.threshold_k - self.sdfs)
        self.hat_k_violated = tf.cast(self.sdfs < self.threshold_k, dtype=tf.int32, name="hat_k_violated")
        _, self.constraint_prediction_accuracy = tf.metrics.accuracy(labels=self.k_label,
                                                                     predictions=self.hat_k_violated)

        with tf.name_scope("train"):
            # sum of squared errors in latent space at each time step
            self.constraint_prediction_error = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.hat_k,
                                                                                       labels=self.k_label,
                                                                                       name='constraint_prediction_err')
            self.constraint_prediction_loss = tf.reduce_mean(self.constraint_prediction_error,
                                                             name="constraint_prediction_loss")

            self.flat_weights = tf.reshape(self.R_k, [-1])
            self.regularization = tf.nn.l2_loss(self.flat_weights) * self.beta

            self.loss = tf.add_n([
                self.constraint_prediction_loss,
                self.regularization
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
            tf.summary.scalar("regularization_loss_summary", self.regularization)
            tf.summary.scalar("loss_summary", self.loss)

            self.summaries = tf.summary.merge_all()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            if args['debug']:
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.saver = tf.train.Saver(max_to_keep=None)

    def train(self, train_x, epochs, log_path):
        interrupted = False

        writer = None
        loss = None
        full_log_path = None
        if self.args['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)

            link_bot_notebooks.experiments_util.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "metadata.json")
            metadata_file = open(metadata_path, 'w')
            metadata = {
                'tf_version': str(tf.__version__),
                'log path': full_log_path,
                'seed': self.args['seed'],
                'checkpoint': self.args['checkpoint'],
                'N': self.N,
                'beta': self.beta,
                'commandline': self.args['commandline'],
            }
            metadata_file.write(json.dumps(metadata, indent=2))

            writer = tf.summary.FileWriter(full_log_path)
            writer.add_graph(self.sess.graph)

        try:
            observations = train_x['states'].reshape(-1, self.N)
            k = train_x['constraints'].reshape(-1, 1)
            feed_dict = {self.observations: observations,
                         self.k_label: k,
                         }

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
            ops = [self.R_k]
            R_k, = self.sess.run(ops, feed_dict={})
            if self.args['verbose']:
                print("Loss: {}".format(loss))
                print("R_k:\n{}".format(R_k))

        return interrupted

    def evaluate(self, eval_x, display=True):
        observations = eval_x['states'].reshape(-1, self.N)
        k = eval_x['constraints'].reshape(-1, 1)
        feed_dict = {self.observations: observations,
                     self.k_label: k,
                     }
        ops = [self.R_k, self.threshold_k, self.constraint_prediction_loss, self.regularization, self.loss,
               self.constraint_prediction_accuracy]
        R_k, threshold_k, k_loss, reg, loss, k_accuracy = self.sess.run(ops, feed_dict=feed_dict)

        if display:
            print("Constraint Loss: {:0.3f}".format(float(k_loss)))
            print("Regularization: {:0.3f}".format(float(reg)))
            print("Overall Loss: {:0.3f}".format(float(loss)))
            print("R_k:\n{}".format(R_k))
            print("threshold_k:\n{}".format(threshold_k))
            print("constraint prediction accuracy:\n{}".format(k_accuracy))

        return R_k, threshold_k, k_loss, reg, loss

    def setup(self):
        if self.args['checkpoint']:
            self.sess.run([tf.local_variables_initializer()])
            self.load()
        else:
            self.init()

    def init(self):
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def constraint_violated(self, latent_k):
        full_latent_k = np.ndarray((1, self.P))
        full_latent_k[0, 0] = latent_k
        feed_dict = {self.hat_latent_k: full_latent_k}
        ops = [self.hat_k_violated]
        constraint_violated = self.sess.run(ops, feed_dict=feed_dict)
        # take the first op from the list, then take the first batch and first time step from that
        constraint_violated = constraint_violated[0, 0]
        return constraint_violated

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

    def __str__(self):
        ops = [self.R_k, self.threshold_k]
        return "R_k:\n{}\nthreshold_k:\n{}\n".format(*self.sess.run(ops, feed_dict={}))
