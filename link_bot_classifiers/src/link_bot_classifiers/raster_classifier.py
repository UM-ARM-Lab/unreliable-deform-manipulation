#!/usr/bin/env python
from __future__ import print_function

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore, Style
from tensorflow.python.training.checkpointable.data_structures import NoDependency

from link_bot_planning.my_motion_validator import MotionClassifier
from link_bot_planning.visualization import plot_classifier_data
from link_bot_pycommon import experiments_util, link_bot_sdf_utils
from moonshine.raster_points_layer import RasterPoints


class RasterClassifier(tf.keras.Model):

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = NoDependency(hparams)
        self.m_dim = self.hparams['n_control']

        self.raster = RasterPoints(self.hparams['sdf_shape'])
        self.conv1 = layers.Conv2D(16, [5, 5], activation='relu')
        self.pool1 = layers.MaxPool2D(2)
        self.conv2 = layers.Conv2D(8, [3, 3], activation='relu')
        self.pool2 = layers.MaxPool2D(2)
        self.conv_flatten = layers.Flatten()
        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(16, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, input_dict, training=None, mask=None):
        planned_state = input_dict['planned_state']
        planned_next_state = input_dict['planned_next_state']
        planned_sdf = input_dict['planned_sdf/sdf']
        planned_sdf_resolution = input_dict['res']
        planned_sdf_origin = input_dict['planned_sdf/origin']

        # add channel index
        planned_sdf = tf.expand_dims(planned_sdf, axis=3)

        # add time index into everything
        planned_state = tf.expand_dims(planned_state, axis=1)
        planned_next_state = tf.expand_dims(planned_next_state, axis=1)
        planned_sdf_origin = tf.expand_dims(planned_sdf_origin, axis=1)
        # convert from (batch, 1, 1) -> (batch, 1, 2)
        planned_sdf_resolution = tf.tile(tf.expand_dims(planned_sdf_resolution, axis=1), [1, 1, 2])

        # raster each state into an image
        planned_rope_image = self.raster([planned_state, planned_sdf_resolution, planned_sdf_origin])
        planned_next_rope_image = self.raster([planned_next_state, planned_sdf_resolution, planned_sdf_origin])

        # remove time index
        image_shape = [planned_rope_image.shape[0],
                       planned_rope_image.shape[2],
                       planned_rope_image.shape[3],
                       planned_rope_image.shape[4]]
        planned_rope_image = tf.reshape(planned_rope_image, image_shape)
        planned_next_rope_image = tf.reshape(planned_next_rope_image, image_shape)

        # batch, h, w, channel
        concat_image = tf.concat((planned_rope_image, planned_next_rope_image, planned_sdf), axis=3)

        # feed into a CNN
        conv_h1 = self.conv1(concat_image)
        conv_z1 = self.pool1(conv_h1)
        conv_h2 = self.conv2(conv_z1)
        conv_z2 = self.pool2(conv_h2)
        conv_output = self.conv_flatten(conv_z2)
        h1 = self.dense1(conv_output)
        h2 = self.dense2(h1)
        out_h = h2

        accept_probability = self.output_layer(out_h)
        return planned_rope_image, planned_next_rope_image, planned_sdf, accept_probability


def eval(hparams, test_tf_dataset, args):
    net = RasterClassifier(hparams=hparams)
    accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    ckpt = tf.train.Checkpoint(net=net)
    # TODO: add accuracy metrics
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)

    loss = tf.keras.losses.BinaryCrossentropy()

    test_losses = []
    accuracy.reset_states()
    for test_example_dict in test_tf_dataset:
        test_labels = test_example_dict['label']
        test_predictions = net(test_example_dict)[-1]
        batch_test_loss = loss(y_true=test_labels, y_pred=test_predictions)
        accuracy.update_state(y_true=test_labels, y_pred=test_predictions)
        test_losses.append(batch_test_loss)
    test_loss = np.mean(test_losses)
    test_accuracy = accuracy.result().numpy() * 100
    print("Test Loss:     {:7.4f}".format(test_loss))
    print("Test Accuracy: {:5.3f}%".format(test_accuracy))


def train(hparams, train_tf_dataset, val_tf_dataset, log_path, args):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    loss = tf.keras.losses.BinaryCrossentropy()
    accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    batch_accuracy = tf.keras.metrics.BinaryAccuracy(name='batch_accuracy')
    net = RasterClassifier(hparams=hparams)
    global_step = tf.train.get_or_create_global_step()

    # If we're resuming a checkpoint, there is no new log path
    if args.checkpoint is not None:
        full_log_path = args.checkpoint
    elif args.log:
        full_log_path = pathlib.Path("log_data") / log_path

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
            hparams_file.write(json.dumps(hparams, indent=2))

        writer = tf.contrib.summary.create_file_writer(logdir=full_log_path)

    def train_loop():
        for epoch in range(args.epochs):
            ################
            # train
            ################
            # metrics are averaged across batches in the epoch
            batch_losses = []
            accuracy.reset_states()
            for train_example_dict_batch in progressbar.progressbar(train_tf_dataset):
                step = global_step.numpy()
                train_true_labels_batch = train_example_dict_batch['label']

                with tf.GradientTape() as tape:
                    fwd_result = net(train_example_dict_batch)
                    i1, i2, sdf, train_predictions_batch = fwd_result
                    training_batch_loss = loss(y_true=train_true_labels_batch, y_pred=train_predictions_batch)
                variables = net.trainable_variables
                gradients = tape.gradient(training_batch_loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                batch_losses.append(training_batch_loss.numpy())
                accuracy.update_state(y_true=train_true_labels_batch, y_pred=train_predictions_batch)

                if args.log:
                    if step % args.log_grad_every == 0:
                        for grad, var in zip(gradients, variables):
                            tf.contrib.summary.histogram(var.name + '_grad', grad, step=step)
                    if step % args.log_scalars_every == 0:
                        batch_accuracy.reset_states()
                        batch_accuracy.update_state(y_true=train_true_labels_batch, y_pred=train_predictions_batch)
                        tf.contrib.summary.scalar('batch accuracy', batch_accuracy.result(), step=step)
                        tf.contrib.summary.scalar("batch loss", training_batch_loss, step=step)

                ####################
                # Update global step
                ####################
                global_step.assign_add(1)

                if args.verbose >= 4:
                    plt.figure()
                    bin = np.tile(sdf[0].numpy(), [1, 1, 3]) * 1.0
                    i1 = i1[0].numpy()
                    i2 = i2[0].numpy()
                    i1_mask = np.tile(i1.sum(axis=2, keepdims=True) > 0, [1, 1, 3])
                    i2_mask = np.tile(i2.sum(axis=2, keepdims=True) > 0, [1, 1, 3])
                    mask = (1 - np.logical_or(i1_mask, i2_mask).astype(np.float32))
                    masked = bin * mask
                    new_image = masked + i1 + i2
                    plt.imshow(new_image)
                    plt.title(train_true_labels_batch[0].numpy()[0])
                    plt.show()

            training_loss = np.mean(batch_losses)
            training_accuracy = accuracy.result().numpy() * 100
            log_msg = "Epoch: {:5d}, Training Loss: {:7.4f}, Training Accuracy: {:5.2f}%"
            print(log_msg.format(epoch, training_loss, training_accuracy))

            ################
            # validation
            ################
            if args.log and epoch % args.validation_every == 0:
                val_losses = []
                accuracy.reset_states()
                for val_example_dict_batch in val_tf_dataset:
                    val_true_labels_batch = val_example_dict_batch['label']
                    val_predictions_batch = net(val_example_dict_batch)[-1]
                    val_loss_batch = loss(y_true=val_true_labels_batch, y_pred=val_predictions_batch)
                    accuracy.update_state(y_true=val_true_labels_batch, y_pred=val_predictions_batch)
                    val_losses.append(val_loss_batch)
                val_loss = np.mean(val_losses)
                val_accuracy = accuracy.result().numpy() * 100
                tf.contrib.summary.scalar('validation loss', val_loss, step=int(ckpt.step))
                tf.contrib.summary.scalar('validation accuracy', val_accuracy, step=int(ckpt.step))
                format_message = "Validation Loss: " + Style.BRIGHT + "{:7.4f}" + Style.RESET_ALL
                format_message += " Accuracy: " + Style.BRIGHT + "{:5.3f}%" + Style.RESET_ALL
                print(format_message.format(val_loss, val_accuracy) + Style.RESET_ALL)

            ################
            # Checkpoint
            ################
            if args.log and epoch % args.save_freq == 0:
                save_path = manager.save()
                print(Fore.CYAN + "Step {:6d}: Saved checkpoint {}".format(int(ckpt.step), save_path) + Fore.RESET)

        if args.log:
            save_path = manager.save()
            print(Fore.CYAN + "Step {:6d}: Saved final checkpoint {}".format(int(ckpt.step), save_path) + Fore.RESET)

    if args.log:
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            train_loop()
    else:
        train_loop()


class RasterClassifierWrapper(MotionClassifier):

    def __init__(self, path: pathlib.Path):
        super().__init__()
        model_hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.net = RasterClassifier(hparams=self.model_hparams)
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)
        self.ckpt.restore(self.manager.latest_checkpoint)
        self.n_control = 2

    def predict(self, local_sdf_data: link_bot_sdf_utils.SDF, s1: np.ndarray, s2: np.ndarray):
        """
        :param s1: [batch, 6] float64
        :param s2: [batch, 6] float64
        :return: [batch, 1] float64
        """
        test_x = {
            'planned_state': tf.convert_to_tensor(s1, dtype=tf.float32),
            'planned_next_state': tf.convert_to_tensor(s2, dtype=tf.float32),
            # TODO: consider making this binary occupancy grid instead of SDF
            'planned_sdf/sdf': tf.convert_to_tensor(local_sdf_data.sdf, dtype=tf.float32),
            'res': tf.convert_to_tensor(local_sdf_data.resolution[0]),
            'planned_sdf/origin': tf.convert_to_tensor(local_sdf_data.origin, dtype=tf.int64),
        }
        accept_probabilities = self.net(test_x)[-1]
        accept_probabilities = accept_probabilities.numpy()
        accept_probabilities = accept_probabilities.as_type(np.float64)

        print(accept_probabilities)
        title = "p(accept) = {}".format(accept_probabilities)
        plot_classifier_data(planned_env=test_x['planned_sdf/sdf'],
                             planned_env_extent=test_x['planned_sdf/extent'],
                             planned_state=test_x['planned_state'],
                             planned_next_state=test_x['planned_next_state'],
                             actual_env=None,
                             actual_env_extent=None,
                             state=None,
                             next_state=None,
                             title=title,
                             label=None)
        plt.show()

        return accept_probabilities
