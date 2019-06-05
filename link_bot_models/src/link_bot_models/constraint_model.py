#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import json
import os
from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import link_bot_pycommon.experiments_util
from link_bot_models.tf_signed_distance_field_op import sdf_func


class ConstraintModelType(Enum):
    FullLinear = 1
    LinearCombination = 2
    FNN = 3

    @classmethod
    def strings(cls):
        return [e.name for e in cls]


class ConstraintModel:

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

        model_type = ConstraintModelType[args['model_type']]
        if model_type == ConstraintModelType.FullLinear:
            ##############################################
            #             Full Linear Model              #
            ##############################################
            if args['random_init']:
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
            if args['random_init']:
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

            self.train_summary = tf.summary.merge_all('train')
            self.validation_summary = tf.summary.merge_all('validation')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        if args['debug']:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.saver = tf.train.Saver(max_to_keep=None)

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

    def train(self, train_observations, train_k, validation_observations, validation_k, epochs, log_path):
        interrupted = False

        writer = None
        loss = None
        full_log_path = None
        if self.args['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)

            link_bot_pycommon.experiments_util.make_log_dir(full_log_path)

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
                'hidden_layer_dims': self.hidden_layer_dims,
                'model_type': self.args['model_type'],
            }
            metadata_file.write(json.dumps(metadata, indent=2))

            writer = tf.summary.FileWriter(full_log_path)
            writer.add_graph(self.sess.graph)

        try:
            train_ops = [self.global_step, self.train_summary, self.loss, self.opt]
            validation_ops = [self.validation_summary, self.loss]
            for i in range(epochs):

                batch_idx = np.random.choice(np.arange(train_observations.shape[0]), size=self.args['batch_size'])
                train_observations_batch = train_observations[batch_idx]
                train_k_batch = train_k[batch_idx]

                train_feed_dict = {self.observations: train_observations_batch,
                                   self.k_label: train_k_batch}
                validation_feed_dict = {self.observations: validation_observations,
                                        self.k_label: validation_k}
                step, train_summary, train_loss, _ = self.sess.run(train_ops, feed_dict=train_feed_dict)
                validation_summary, validation_loss = self.sess.run(validation_ops, feed_dict=validation_feed_dict)

                if 'save_period' in self.args and (step % self.args['save_period'] == 0 or step == 1):
                    if self.args['log'] is not None:
                        writer.add_summary(train_summary, step)
                        writer.add_summary(validation_summary, step)
                        self.save(full_log_path, loss=validation_loss)

                if 'print_period' in self.args and (step % self.args['print_period'] == 0 or step == 1):
                    print('step: {:4d}, train loss: {:8.4f} val loss {:8.4f}'.format(step, train_loss, validation_loss))

        except KeyboardInterrupt:
            print("stop!!!")
            interrupted = True
            pass
        finally:
            if self.args['verbose']:
                print("Loss: {}".format(loss))

        return interrupted

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
