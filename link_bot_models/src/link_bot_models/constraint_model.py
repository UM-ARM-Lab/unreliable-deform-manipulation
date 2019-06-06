#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

from enum import auto

import numpy as np
import tensorflow as tf

from link_bot_models.base_model import BaseModel
from link_bot_models.tf_signed_distance_field_op import sdf_func
from link_bot_pycommon import link_bot_pycommon


class ConstraintModelType(link_bot_pycommon.ArgsEnum):
    FullLinear = auto()
    LinearCombination = auto()
    FNN = auto()


class ConstraintModel(BaseModel):

    def __init__(self, args, np_sdf, np_sdf_gradient, np_sdf_resolution, np_sdf_origin, N):
        super(ConstraintModel, self).__init__(args, N)

        self.beta = 1e-8

        self.observations = tf.placeholder(tf.float32, shape=(None, N), name="observations")
        self.k_label = tf.placeholder(tf.float32, shape=(None, 1), name="k")
        self.k_label_int = tf.cast(self.k_label, tf.int32)
        self.hidden_layer_dims = None

        self.sdf = np_sdf
        self.sdf_resolution = np_sdf_resolution
        self.sdf_origin = np_sdf_origin

        model_type = self.args_dict['model_type']
        if model_type == ConstraintModelType.FullLinear:
            ##############################################
            #             Full Linear Model              #
            ##############################################
            if self.args_dict['random_init']:
                # RANDOM INIT
                R_k_init = np.random.randn(N, 2).astype(np.float32) * 1e-1
                k_threshold_init = np.random.rand() * 1e-1
            else:
                # IDEAL INIT
                R_k_init = np.zeros((N, 2), dtype=np.float32) + np.random.randn(N, 2).astype(np.float32) * 0.1
                R_k_init[N - 2, 0] = 1.0
                R_k_init[N - 1, 1] = 1.0
                k_threshold_init = 0.20
            self.R_k = tf.get_variable("R_k", initializer=R_k_init)
            self.threshold_k = tf.get_variable("threshold_k", initializer=k_threshold_init, trainable=True)
            self.hat_latent_k = tf.matmul(self.observations, self.R_k, name='hat_latent_k')

        elif model_type == ConstraintModelType.LinearCombination:
            ################################################
            #           Linear Combination Model           #
            ################################################
            if self.args_dict['random_init']:
                # RANDOM INIT
                alphas_init = np.random.randn(3).astype(np.float32)
                k_threshold_init = np.random.rand() * 1e-1
            else:
                # IDEAL INIT
                alphas_init = np.array([0, 1, 1]).astype(np.float32)
                k_threshold_init = 0.20
            self.alphas = tf.get_variable("R_k", initializer=alphas_init)
            alpha_blocks = []
            for alpha in tf.unstack(self.alphas):
                alpha_blocks.append(tf.linalg.tensor_diag([alpha, alpha]))
            self.alpha_blocks = tf.concat(alpha_blocks, axis=0)
            self.R_k = tf.stack(self.alpha_blocks, axis=0)
            self.threshold_k = tf.get_variable("threshold_k", initializer=k_threshold_init, trainable=True)
            self.hat_latent_k = tf.matmul(self.observations, self.R_k, name='hat_latent_k')

        elif model_type == ConstraintModelType.FNN:
            #############################################
            #      Feed-Forward Neural Network Model    #
            #############################################
            k_threshold_init = 0.20
            self.threshold_k = tf.get_variable("threshold_k", initializer=k_threshold_init, trainable=False)
            self.hidden_layer_dims = [128, 128]
            h = self.observations
            for layer_idx, hidden_layer_dim in enumerate(self.hidden_layer_dims):
                h = tf.layers.dense(h, hidden_layer_dim, activation=tf.nn.relu,
                                    name='hidden_layer_{}'.format(layer_idx))
            self.hat_latent_k = tf.layers.dense(h, 2, activation=None, name='output_layer')

        #######################################################
        #                 End Model Definition                #
        #######################################################

        self.sdfs = sdf_func(np_sdf, np_sdf_gradient, np_sdf_resolution, np_sdf_origin, self.hat_latent_k, 2)
        # because the sigmoid is not very sharp (since meters are very large in the domain of sigmoid),
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

            self.loss = tf.add_n([
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

            tf.summary.scalar("constraint_prediction_accuracy_summary", self.constraint_prediction_accuracy,
                              collections=['train'])
            tf.summary.scalar("k_threshold_summary", self.threshold_k, collections=['train'])
            tf.summary.scalar("constraint_prediction_loss_summary", self.constraint_prediction_loss,
                              collections=['train'])
            tf.summary.scalar("loss_summary", self.loss,
                              collections=['train'])

        with tf.name_scope("validation"):
            tf.summary.scalar("constraint_prediction_accuracy_summary", self.constraint_prediction_accuracy,
                              collections=['validation'])
            tf.summary.scalar("k_threshold_summary", self.threshold_k, collections=['validation'])
            tf.summary.scalar("constraint_prediction_loss_summary", self.constraint_prediction_loss,
                              collections=['validation'])
            tf.summary.scalar("loss_summary", self.loss,
                              collections=['validation'])

        self.finish_setup()

    def split_data(self, full_x, fraction_validation=0.10):
        full_observations = full_x['states'].reshape(-1, self.N)
        full_k = full_x['constraints'].reshape(-1, 1)

        end_train_idx = int(full_observations.shape[0] * (1 - fraction_validation))
        shuffled_idx = np.random.permutation(full_observations.shape[0])
        full_observations = full_observations[shuffled_idx]
        full_k = full_k[shuffled_idx]
        train_observations = full_observations[:end_train_idx]
        validation_observations = full_observations[end_train_idx:]
        train_k = full_k[:end_train_idx]
        validation_k = full_k[end_train_idx:]
        return train_observations, train_k, validation_observations, validation_k

    def metadata(self):
        metadata = {
            'tf_version': str(tf.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'beta': self.beta,
            'commandline': self.args_dict['commandline'],
            'hidden_layer_dims': self.hidden_layer_dims,
            'model_type': self.args_dict['model_type'],
        }
        return metadata

    def build_feed_dict(self, x, y):
        return {self.observations: x,
                self.k_label: y}

    def evaluate(self, observations, k, display=True):
        feed_dict = {self.observations: observations,
                     self.k_label: k}
        ops = [self.threshold_k, self.constraint_prediction_loss, self.loss, self.constraint_prediction_accuracy]
        threshold_k, k_loss, loss, k_accuracy = self.sess.run(ops, feed_dict=feed_dict)

        hat_latent_k = self.sess.run(self.hat_latent_k, feed_dict=feed_dict)

        import matplotlib.pyplot as plt
        from PIL import Image
        plt.figure()
        skip = 20
        o_scatter_x = observations[::skip, 4]
        o_scatter_y = observations[::skip, 5]
        h_scatter_x = hat_latent_k[::skip, 0]
        h_scatter_y = hat_latent_k[::skip, 1]
        img = Image.fromarray(np.uint8(np.flipud(self.sdf.T) > threshold_k))
        small_sdf = img.resize((50, 50))
        plt.imshow(small_sdf, extent=[-5, 5, -5, 5])
        for ox, oy, hx, hy in zip(o_scatter_x, o_scatter_y, h_scatter_x, h_scatter_y):
            plt.plot([ox, hx], [oy, hy], c='k', linewidth=1, zorder=1)
        plt.scatter(o_scatter_x, o_scatter_y, s=25, c='blue', zorder=2)
        plt.scatter(h_scatter_x, h_scatter_y, s=25, c='red', zorder=2)
        plt.show()

        if display:
            print("Constraint Loss: {:0.3f}".format(float(k_loss)))
            print("Overall Loss: {:0.3f}".format(float(loss)))
            print("threshold_k:\n{}".format(threshold_k))
            print("constraint prediction accuracy:\n{}".format(k_accuracy))

        return threshold_k, k_loss, loss

    def violated(self, observation, sdf=None, sdf_resolution=None, sdf_origin=None):
        # unused parameters
        del sdf, sdf_resolution, sdf_origin
        feed_dict = {self.observations: np.atleast_2d(observation)}
        violated, pt = self.sess.run([self.hat_k_violated, self.hat_latent_k], feed_dict=feed_dict)
        return np.any(violated), pt

    def constraint_violated(self, latent_k):
        full_latent_k = np.ndarray((1, self.P))
        full_latent_k[0, 0] = latent_k
        feed_dict = {self.hat_latent_k: full_latent_k}
        ops = [self.hat_k_violated]
        constraint_violated = self.sess.run(ops, feed_dict=feed_dict)
        # take the first op from the list, then take the first batch and first time step from that
        constraint_violated = constraint_violated[0, 0]
        return constraint_violated

    def __str__(self):
        ops = [self.threshold_k]
        return "threshold_k:\n{}\n".format(*self.sess.run(ops, feed_dict={}))