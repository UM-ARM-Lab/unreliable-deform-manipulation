#!/usr/bin/env python
from __future__ import print_function

import json
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore, Style
from tensorflow import keras

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_planning.params import LocalEnvParams
from link_bot_pycommon import experiments_util, link_bot_sdf_utils
from moonshine.raster_points_layer import RasterPoints


class RasterClassifier(tf.keras.Model):

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = tf.contrib.checkpoint.NoDependency(hparams)
        self.dynamics_dataset_hparams = self.hparams['classifier_dataset_hparams']['fwd_model_hparams'][
            'dynamics_dataset_hparams']
        self.m_dim = self.dynamics_dataset_hparams['n_action']

        local_env_params = LocalEnvParams.from_json(self.dynamics_dataset_hparams['local_env_params'])
        self.raster = RasterPoints([local_env_params.h_rows, local_env_params.w_cols])
        self.conv_layers = []
        self.pool_layers = []
        for n_filters, kernel_size in self.hparams['conv_filters']:
            conv = layers.Conv2D(n_filters,
                                 kernel_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 activity_regularizer=keras.regularizers.l1(self.hparams['activity_reg']))
            pool = layers.MaxPool2D(2)
            self.conv_layers.append(conv)
            self.pool_layers.append(pool)

        self.conv_flatten = layers.Flatten()
        self.batch_norm = layers.BatchNormalization()

        self.dense_layers = []
        self.dropout_layers = []
        for hidden_size in self.hparams['fc_layer_sizes']:
            dropout = layers.Dropout(rate=self.hparams['dropout_rate'])
            dense = layers.Dense(hidden_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 activity_regularizer=keras.regularizers.l1(self.hparams['activity_reg']))
            self.dropout_layers.append(dropout)
            self.dense_layers.append(dense)

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, input_dict: dict, training=None, mask=None):
        """
        we expect res to be [batch_size, 1]
        """
        planned_state = input_dict['planned_state']
        planned_next_state = input_dict['planned_state_next']
        planned_local_env = input_dict['planned_local_env/env']
        planned_local_env_resolution = input_dict['resolution']
        planned_local_env_origin = input_dict['planned_local_env/origin']

        # add channel index
        planned_local_env = tf.expand_dims(planned_local_env, axis=3)

        # add time index into everything
        planned_state = tf.expand_dims(planned_state, axis=1)
        planned_next_state = tf.expand_dims(planned_next_state, axis=1)
        planned_local_env_origin = tf.expand_dims(planned_local_env_origin, axis=1)
        # convert from (batch, 1, 1) -> (batch, 1, 2)
        planned_local_env_resolution = tf.tile(tf.expand_dims(planned_local_env_resolution, axis=1), [1, 1, 2])

        # raster each state into an image
        planned_rope_image = self.raster([planned_state, planned_local_env_resolution, planned_local_env_origin])
        planned_next_rope_image = self.raster([planned_next_state, planned_local_env_resolution, planned_local_env_origin])

        # remove time index
        image_shape = [planned_rope_image.shape[0],
                       planned_rope_image.shape[2],
                       planned_rope_image.shape[3],
                       planned_rope_image.shape[4]]
        planned_rope_image = tf.reshape(planned_rope_image, image_shape)
        planned_next_rope_image = tf.reshape(planned_next_rope_image, image_shape)

        # batch, h, w, channel
        concat_image = tf.concat((planned_rope_image, planned_next_rope_image, planned_local_env), axis=3)

        # feed into a CNN
        conv_z = concat_image
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z

        conv_output = self.conv_flatten(out_conv_z)

        if self.hparams['batch_norm']:
            conv_output = self.batch_norm(conv_output)

        z = conv_output
        for dropout_layer, dense_layer in zip(self.dropout_layers, self.dense_layers):
            h = dropout_layer(z)
            z = dense_layer(h)
        out_h = z

        accept_probability = self.output_layer(out_h)
        return planned_rope_image, planned_next_rope_image, planned_local_env, accept_probability


def check_validation(val_tf_dataset, loss, net):
    val_losses = []
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    val_accuracy.reset_states()
    for val_example_dict_batch in val_tf_dataset:
        val_true_labels_batch = val_example_dict_batch['label']
        val_predictions_batch = net(val_example_dict_batch)[-1]
        val_loss_batch = loss(y_true=val_true_labels_batch, y_pred=val_predictions_batch)
        val_accuracy.update_state(y_true=val_true_labels_batch, y_pred=val_predictions_batch)
        val_losses.append(val_loss_batch)
    val_losses = np.array(val_losses)
    return val_losses, val_accuracy


def train(hparams, train_tf_dataset, val_tf_dataset, log_path, args):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    loss = tf.keras.losses.BinaryCrossentropy()
    train_epoch_accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    batch_accuracy = tf.keras.metrics.BinaryAccuracy(name='batch_accuracy')
    net = RasterClassifier(hparams=hparams)
    global_step = tf.train.get_or_create_global_step()

    # If we're resuming a checkpoint, there is no new log path
    if args.checkpoint is not None:
        full_log_path = args.checkpoint
    elif args.log:
        full_log_path = pathlib.Path("log_data") / log_path
    else:
        full_log_path = '/tmp'

    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, net=net)
    manager = tf.train.CheckpointManager(ckpt, full_log_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
    elif args.checkpoint:
        print(Fore.RED + "Failed to restore from checkpoint directory {}".format(args.checkpoint) + Fore.RESET)
        print("Did you forget a subdirectory?")
        return

    writer = None
    if args.log is not None:
        print(Fore.CYAN + "Logging to {}".format(full_log_path) + Fore.RESET)

        experiments_util.make_log_dir(full_log_path)

        hparams_path = full_log_path / "hparams.json"
        with hparams_path.open('w') as hparams_file:
            hparams['log path'] = str(full_log_path)
            hparams['dataset'] = str(args.dataset_dirs)
            hparams_file.write(json.dumps(hparams, indent=2))

        writer = tf.contrib.summary.create_file_writer(logdir=full_log_path)

    def train_loop():
        for epoch in range(args.epochs):
            ################
            # train
            ################
            # metrics are averaged across batches in the epoch
            batch_losses = []
            train_epoch_accuracy.reset_states()
            for train_example_dict_batch in progressbar.progressbar(train_tf_dataset):
                step = int(global_step.numpy())
                train_true_labels_batch = train_example_dict_batch['label']

                with tf.GradientTape() as tape:
                    fwd_result = net(train_example_dict_batch)
                    i1, i2, local_env, train_predictions_batch = fwd_result
                    training_batch_loss = loss(y_true=train_true_labels_batch, y_pred=train_predictions_batch)
                variables = net.trainable_variables
                gradients = tape.gradient(training_batch_loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                batch_losses.append(training_batch_loss.numpy())
                train_epoch_accuracy.update_state(y_true=train_true_labels_batch, y_pred=train_predictions_batch)

                if args.log:
                    if step % args.log_grad_every == 0:
                        for grad, var in zip(gradients, variables):
                            tf.contrib.summary.histogram(var.name + '_grad', grad, step=step)
                    if step % args.log_scalars_every == 0:
                        batch_accuracy.reset_states()
                        batch_accuracy.update_state(y_true=train_true_labels_batch, y_pred=train_predictions_batch)
                        tf.contrib.summary.scalar('batch accuracy', batch_accuracy.result(), step=step)
                        tf.contrib.summary.scalar("batch loss", training_batch_loss, step=step)
                    ################
                    # validation
                    ################
                    if step % args.validation_every == 0:
                        val_losses, val_accuracy = check_validation(val_tf_dataset, loss, net)
                        mean_val_loss = np.mean(val_losses)
                        val_accuracy = val_accuracy.result().numpy() * 100
                        tf.contrib.summary.scalar('validation loss', mean_val_loss, step=step)
                        tf.contrib.summary.scalar('validation accuracy', val_accuracy, step=step)
                        format_message = "Validation Loss: " + Style.BRIGHT + "{:7.4f}" + Style.RESET_ALL
                        format_message += " Accuracy: " + Style.BRIGHT + "{:5.3f}%" + Style.RESET_ALL
                        print(format_message.format(mean_val_loss, val_accuracy) + Style.RESET_ALL)

                ####################
                # Update global step
                ####################
                global_step.assign_add(1)

                if args.verbose >= 4:
                    plt.figure()
                    bin = np.tile(local_env[0].numpy(), [1, 1, 3]) * 1.0
                    i1 = i1[0].numpy()
                    i2 = i2[0].numpy()
                    i1_mask = np.tile(i1.sum(axis=2, keepdims=True) > 0, [1, 1, 3])
                    i2_mask = np.tile(i2.sum(axis=2, keepdims=True) > 0, [1, 1, 3])
                    mask = (1 - np.logical_or(i1_mask, i2_mask).astype(np.float32))
                    # mask = (1 - i1_mask.astype(np.float32))
                    masked = bin * mask
                    new_image = masked + i1
                    plt.imshow(new_image)
                    plt.title(train_true_labels_batch[0].numpy()[0])
                    plt.show()

            training_loss = np.mean(batch_losses)
            training_accuracy = train_epoch_accuracy.result().numpy() * 100
            log_msg = "Epoch: {:5d}, Training Loss: {:7.4f}, Training Accuracy: {:5.2f}%"
            print(log_msg.format(epoch, training_loss, training_accuracy))

            ################
            # Checkpoint
            ################
            if args.log and epoch % args.save_freq == 0:
                save_path = manager.save()
                print(Fore.CYAN + "Step {:6d}: Saved checkpoint {}".format(step, save_path) + Fore.RESET)

        if args.log:
            save_path = manager.save()
            print(Fore.CYAN + "Step {:6d}: Saved final checkpoint {}".format(step, save_path) + Fore.RESET)

    if args.log:
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            train_loop()
    else:
        train_loop()


def eval(hparams, test_tf_dataset, args):
    net = RasterClassifier(hparams=hparams)
    accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    ckpt = tf.train.Checkpoint(net=net)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)

    loss = tf.keras.losses.BinaryCrossentropy()

    test_losses = []
    accuracy.reset_states()
    test_predictions = []
    tn = 0
    fn = 0
    fp = 0
    tp = 0
    for test_example_dict in test_tf_dataset:
        test_batch_labels = test_example_dict['label']
        test_batch_predictions = net(test_example_dict)[-1]
        test_predictions.append(test_batch_predictions.numpy().flatten())
        batch_test_loss = loss(y_true=test_batch_labels, y_pred=test_batch_predictions)
        accuracy.update_state(y_true=test_batch_labels, y_pred=test_batch_predictions)
        test_losses.append(batch_test_loss)

        for pred, label in zip(test_batch_predictions, test_batch_labels):
            pred_bin = int(pred.numpy() > 0.5)
            label = label.numpy()[0]
            if pred_bin == 1 and label == 1:
                tp += 1
            elif pred_bin == 0 and label == 1:
                fn += 1
            elif pred_bin == 1 and label == 0:
                fp += 1
            elif pred_bin == 0 and label == 0:
                tn += 1

    test_loss = np.mean(test_losses)
    test_accuracy = accuracy.result().numpy() * 100
    print("Test Loss:     {:7.4f}".format(test_loss))
    print("Test Accuracy: {:5.3f}%".format(test_accuracy))

    print("|        | label 0 | label 1 |")
    print("| pred 0 | {:6d} | {:6d} |".format(tn, fn))
    print("| pred 1 | {:6d} | {:6d} |".format(fp, tp))


class RasterClassifierWrapper(BaseClassifier):

    def __init__(self, path: pathlib.Path, show: bool = False):
        super().__init__()
        model_hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.net = RasterClassifier(hparams=self.model_hparams)
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        self.ckpt.restore(self.manager.latest_checkpoint)
        self.n_control = 2
        self.show = show

    def predict(self, local_env_data: List, s1_s: np.ndarray, s2_s: np.ndarray) -> float:
        """
        :param local_env_datas:
        :param s1: [batch, 6] float64
        :param s2: [batch, 6] float64
        :return: [batch, 1] float64
        """
        data_s, res_s, origin_s, extent_s = link_bot_sdf_utils.batch_occupancy_data(local_env_data)
        test_x = {
            'planned_state': tf.convert_to_tensor(s1_s, dtype=tf.float32),
            'planned_state_next': tf.convert_to_tensor(s2_s, dtype=tf.float32),
            'planned_local_env/env': tf.convert_to_tensor(data_s, dtype=tf.float32),
            'resolution': tf.convert_to_tensor(res_s, dtype=tf.float32),
            'planned_local_env/origin': tf.convert_to_tensor(origin_s, dtype=tf.float32),
            'planned_local_env/extent': tf.convert_to_tensor(extent_s, dtype=tf.float32),
        }
        accept_probabilities = self.net(test_x)[-1]
        accept_probabilities = accept_probabilities.numpy()
        accept_probabilities = accept_probabilities.astype(np.float64)[:, 0]

        if self.show:
            title = "n_parallel_calls(accept) = {:5.3f}".format(accept_probabilities)
            plot_classifier_data(planned_env=local_env_data[0].data,
                                 planned_env_extent=local_env_data[0].extent,
                                 planned_state=s1_s[0],
                                 planned_next_state=s2_s[0],
                                 actual_env=None,
                                 actual_env_extent=None,
                                 state=None,
                                 next_state=None,
                                 title=title,
                                 label=None)
            plt.show()

        return accept_probabilities
