from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
from colorama import Style
import tensorflow as tf
from attr import dataclass

from link_bot_models import plotting
from link_bot_models.base_model import BaseModel
from link_bot_models.label_types import LabelType
from link_bot_models.tf_signed_distance_field_op import sdf_func
from link_bot_pycommon import link_bot_pycommon


class ConstraintSDF(BaseModel):

    def __init__(self, args, sdf_shape, N):
        super(ConstraintSDF, self).__init__(args, N)

        self.label_type = self.args_dict['label_type']
        if self.label_type == LabelType.SDF:
            self.label_mask = np.array([1, 0], dtype=np.int)
        elif self.label_type == LabelType.Overstretching:
            self.label_mask = np.array([0, 1], dtype=np.int)
        elif self.label_type == LabelType.SDF_and_Overstretching:
            self.label_mask = np.array([1, 1], dtype=np.int)

        self.beta = 1e-2

        batch_dim = None
        self.sdf = tf.placeholder(tf.float32, shape=[batch_dim, sdf_shape[0], sdf_shape[1]], name="sdf")
        self.sdf_gradient = tf.placeholder(tf.float32, shape=[batch_dim, sdf_shape[0], sdf_shape[1], 2], name="sdf_gradient")
        self.sdf_origin = tf.placeholder(tf.int32, shape=[batch_dim, 2], name="sdf_origin")
        self.sdf_resolution = tf.placeholder(tf.float32, shape=[batch_dim, 2], name="sdf_resolution")
        self.sdf_extent = tf.placeholder(tf.float32, shape=[batch_dim, 4], name="sdf_extent")

        self.observations = tf.placeholder(tf.float32, shape=[batch_dim, N], name="observations")
        self.k_label = tf.placeholder(tf.float32, shape=[batch_dim, 1], name="k")
        self.k_label_int = tf.cast(self.k_label, tf.int32)
        self.hidden_layer_dims = None
        self.fig = None

        ##########################
        # Start Model Definition #
        ##########################

        k_threshold_init = 0.0
        self.threshold_k = tf.get_variable("threshold_k", initializer=k_threshold_init, trainable=False)
        self.hidden_layer_dims = [6]
        h = self.observations
        for layer_idx, hidden_layer_dim in enumerate(self.hidden_layer_dims):
            h = tf.layers.dense(h, hidden_layer_dim, activation=tf.nn.relu, use_bias=False,
                                name='hidden_layer_{}'.format(layer_idx))
        self.hat_latent_k = tf.layers.dense(h, 2, activation=None, use_bias=True, name='output_layer')

        #######################################################
        #                 End Model Definition                #
        #######################################################

        self.sdfs = sdf_func(self.sdf, self.sdf_gradient, self.sdf_resolution, self.sdf_origin, self.hat_latent_k, 2)
        self.sigmoid_scale = 1.0
        self.hat_k = self.sigmoid_scale * (self.threshold_k - self.sdfs)
        self.hat_k_violated = tf.cast(self.sdfs < self.threshold_k, dtype=tf.int32, name="hat_k_violated")
        _, self.constraint_prediction_accuracy = tf.metrics.accuracy(labels=self.k_label,
                                                                     predictions=self.hat_k_violated,
                                                                     name="constraint_prediction_accuracy_metric")
        _, self.constraint_prediction_precision = tf.metrics.precision(labels=self.k_label,
                                                                       predictions=self.hat_k_violated,
                                                                       name="constraint_prediction_precision_metric")
        _, self.constraint_prediction_recall = tf.metrics.recall(labels=self.k_label,
                                                                 predictions=self.hat_k_violated,
                                                                 name="constraint_prediction_recall_metric")
        self.accuracy_local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="constraint_prediction_accuracy_metric")
        self.accuracy_local_vars_initializer = tf.variables_initializer(var_list=self.accuracy_local_vars)
        self.precision_local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                                      scope="constraint_prediction_precision_metric")
        self.precision_local_vars_initializer = tf.variables_initializer(var_list=self.precision_local_vars)
        self.recall_local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="constraint_prediction_recall_metric")
        self.recall_local_vars_initializer = tf.variables_initializer(var_list=self.recall_local_vars)

        with tf.name_scope("train"):
            # sum of squared errors in latent space at each time step
            self.constraint_prediction_error = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.hat_k,
                                                                                       labels=self.k_label,
                                                                                       name='constraint_prediction_err')
            self.constraint_prediction_loss = tf.reduce_mean(self.constraint_prediction_error,
                                                             name="constraint_prediction_loss")

            # FIXME: this assumes that the physical world coordinates (0,0) in meters is the origin/center of the SDF
            self.distances_to_origin = tf.norm(self.hat_latent_k, axis=1)
            oob_left = self.hat_latent_k[:, 0] <= self.sdf_extent[:, 0]
            oob_right = self.hat_latent_k[:, 0] >= self.sdf_extent[:, 1]
            oob_up = self.hat_latent_k[:, 1] <= self.sdf_extent[:, 2]
            oob_down = self.hat_latent_k[:, 1] >= self.sdf_extent[:, 3]
            self.out_of_bounds = tf.math.reduce_any(tf.stack((oob_up, oob_down, oob_left, oob_right), axis=1), axis=1, name='oob')
            self.in_bounds_value = tf.ones_like(self.distances_to_origin) * 0.0
            self.distances_out_of_bounds = tf.where(self.out_of_bounds, self.distances_to_origin, self.in_bounds_value)
            self.out_of_bounds_loss = tf.reduce_mean(self.distances_out_of_bounds, name='out_of_bounds_loss')

            self.loss = tf.add_n([
                self.constraint_prediction_loss,
                self.beta * self.out_of_bounds_loss,
            ], name='loss')

            self.global_step = tf.get_variable("global_step", initializer=0, trainable=False)
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, global_step=self.global_step)

            trainable_vars = tf.trainable_variables()
            for var in trainable_vars:
                name = var.name.replace(":", "_")
                grads = tf.gradients(self.loss, var, name='dLoss_d{}'.format(name))
                for grad in grads:
                    if grad is not None:
                        tf.summary.histogram(name + "/gradient", grad, collections=['train'])
                    else:
                        print("Warning... there is no gradient of the loss with respect to {}".format(var.name))

            tf.summary.scalar("constraint_prediction_accuracy_summary", self.constraint_prediction_accuracy,
                              collections=['train'])
            tf.summary.scalar("constraint_prediction_precision_summary", self.constraint_prediction_precision,
                              collections=['train'])
            tf.summary.scalar("constraint_prediction_recall_summary", self.constraint_prediction_recall,
                              collections=['train'])
            # tf.summary.scalar("k_threshold_summary", self.threshold_k, collections=['train'])
            tf.summary.scalar("out_of_bounds_loss_summary", self.out_of_bounds_loss,
                              collections=['train'])
            tf.summary.scalar("constraint_prediction_loss_summary", self.constraint_prediction_loss,
                              collections=['train'])
            tf.summary.scalar("loss_summary", self.loss,
                              collections=['train'])

        with tf.name_scope("validation"):
            tf.summary.scalar("constraint_prediction_accuracy_summary", self.constraint_prediction_accuracy,
                              collections=['validation'])
            tf.summary.scalar("constraint_prediction_precision_summary", self.constraint_prediction_precision,
                              collections=['validation'])
            tf.summary.scalar("constraint_prediction_recall_summary", self.constraint_prediction_recall,
                              collections=['validation'])
            # tf.summary.scalar("k_threshold_summary", self.threshold_k, collections=['validation'])
            tf.summary.scalar("out_of_bounds_loss_summary", self.out_of_bounds_loss,
                              collections=['validation'])
            tf.summary.scalar("constraint_prediction_loss_summary", self.constraint_prediction_loss,
                              collections=['validation'])
            tf.summary.scalar("loss_summary", self.loss,
                              collections=['validation'])

        self.finish_setup()

    def metadata(self):
        metadata = {
            'tf_version': str(tf.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'beta': self.beta,
            'sigmoid_scale': self.sigmoid_scale,
            'label_type': str(self.label_type),
            'commandline': self.args_dict['commandline'],
            'hidden_layer_dims': self.hidden_layer_dims,
        }
        return metadata

    def build_feed_dict(self, x, y: np.ndarray, **kwargs):
        """
        :param x: first dim is type of input, second dim is batch, following dims are data
        :param y: first dim is type of label, second dim is batch, following dims are labels
        :param kwargs:
        :return:
        """
        labels = np.any(y[0] * self.label_mask, axis=1, keepdims=True).astype(np.float32)
        return {self.observations: x[0],
                self.sdf: x[1],
                self.sdf_gradient: x[2],
                self.sdf_origin: x[3],
                self.sdf_resolution: x[4],
                self.sdf_extent: x[5],
                self.k_label: labels}

    def evaluate(self, eval_x, eval_y, display=True):
        feed_dict = self.build_feed_dict(eval_x, eval_y)
        ops = [self.threshold_k, self.constraint_prediction_loss, self.loss, self.constraint_prediction_accuracy,
               self.constraint_prediction_precision, self.constraint_prediction_recall]
        threshold, k_loss, loss, accuracy, precision, recall = self.sess.run(ops, feed_dict=feed_dict)

        if display:
            print("Constraint Loss: {:0.3f}".format(float(k_loss)))
            print("Overall Loss: {:0.3f}".format(float(loss)))
            print("Precision: {:4.1f}%".format(precision * 100))
            print("Recall: {:4.1f}%".format(recall * 100))
            print(Style.BRIGHT + "Accuracy: {:4.1f}%".format(accuracy * 100) + Style.NORMAL)

        return threshold, k_loss, loss

    def start_train_hook(self):
        self.fig = plt.figure()

    def train_init_epoch_hook(self, epoch, step):
        # reset at the start of each epoch we are computing accuracy for the training set but not all training history
        self.sess.run(self.accuracy_local_vars_initializer)

    def validation_init_hook(self, epoch, step):
        # reset at the start of validation so we clear the metrics from training steps
        self.sess.run(self.accuracy_local_vars_initializer)

    def train_feed_hook(self, iteration, train_x_batch, train_y_batch):
        if 'plot_gradient_descent' in self.args_dict:
            if iteration % 10 == 0:
                threshold_k = self.sess.run(self.threshold_k)
                self.fig.clf()
                x = train_x_batch[::10]
                y = train_y_batch[::10]
                plotting.plot_examples_on_fig(self.fig, x, y, threshold_k, self)
                plt.pause(5)

    def violated(self, observations, sdf_data):
        n_observations = observations.shape[0]
        sdfs = np.tile(sdf_data.sdf, [n_observations, 1, 1])
        sdf_origins = np.tile(sdf_data.origin, [n_observations, 1])
        sdf_resolutions = np.tile(sdf_data.resolution, [n_observations, 1])
        sdf_extents = np.tile(sdf_data.extent, [n_observations, 1])
        feed_dict = {
            self.observations: observations,
            self.sdf: sdfs,
            self.sdf_origin: sdf_origins,
            self.sdf_resolution: sdf_resolutions,
            self.sdf_extent: sdf_extents,
        }
        predicted_violated, pt = self.sess.run([self.hat_k_violated, self.hat_latent_k], feed_dict=feed_dict)
        return np.any(predicted_violated, axis=1), pt

    def __str__(self):
        return "sdf model"


@dataclass
class EvaluateResult:
    rope_configuration: np.ndarray
    predicted_point: np.ndarray
    predicted_violated: bool
    true_violated: bool


def test_single_prediction(sdf_data, model, threshold, rope_configuration):
    rope_configuration = rope_configuration.reshape(-1, 6)
    predicted_violated, predicted_point = model.violated(rope_configuration, sdf_data)
    predicted_point = predicted_point.squeeze()
    rope_configuration = rope_configuration.squeeze()
    head_x = rope_configuration[4]
    head_y = rope_configuration[5]
    row_col = link_bot_pycommon.point_to_sdf_idx(head_x, head_y, sdf_data.resolution, sdf_data.origin)
    true_violated = sdf_data.sdf[row_col] < threshold

    result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, true_violated)
    return result


def test_predictions(model, environment):
    rope_configurations = environment.rope_data['rope_configurations']
    constraint_labels = environment.rope_data['constraints']

    predicted_violateds, predicted_points = model.violated(rope_configurations, environment.sdf_data)

    m = rope_configurations.shape[0]
    results = np.ndarray([m], dtype=EvaluateResult)
    for i in range(m):
        rope_configuration = rope_configurations[i]
        predicted_point = predicted_points[i]
        predicted_violated = predicted_violateds[i]
        constraint_label = constraint_labels[i]
        result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, constraint_label)
        results[i] = result
    return results
