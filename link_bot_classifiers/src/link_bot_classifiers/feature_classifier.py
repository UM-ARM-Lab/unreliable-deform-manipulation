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

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_pycommon import experiments_util, link_bot_sdf_utils
from link_bot_pycommon.link_bot_pycommon import angle_2d_batch_tf


class FeatureClassifier(tf.keras.Model):

    def __init__(self, hparams, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = tf.contrib.checkpoint.NoDependency(hparams)
        self.batch_size = batch_size

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, input_dict: dict, training=None, mask=None):
        """
        Expected sizes:
            'action': n_batch, n_action
            'planned_state': n_batch, n_state
            'planned_state_next': n_batch, n_state
            'planned_local_env/env': n_batch, h, w
            'planned_local_env/origin': n_batch, 2
            'planned_local_env/extent': n_batch, 4
            'resolution': n_batch, 1
        """
        planned_state = input_dict['planned_state']
        action = input_dict['action']
        planned_next_state = input_dict['planned_state_next']
        planned_local_env = input_dict['planned_local_env/env']

        points = tf.reshape(planned_state, [self.batch_size, -1, 2])
        deltas = points[:, 1:] - points[:, :-1]
        distances = tf.norm(deltas, axis=2)
        rope_lengths = tf.reduce_sum(distances, axis=1)

        v1 = tf.reshape(deltas[:, :-1], [-1, 2])
        v2 = tf.reshape(deltas[:, 1:], [-1, 2])
        angles = tf.reshape(angle_2d_batch_tf(v1, v2), [self.batch_size, -1])
        wiggle = tf.reduce_mean(tf.abs(angles), axis=1)

        next_points = tf.reshape(planned_next_state, [self.batch_size, -1, 2])
        next_distances = tf.norm(next_points[:, 1:] - next_points[:, -1:], axis=2)
        next_rope_lengths = tf.reduce_sum(next_distances, axis=1)

        occ = tf.reduce_sum(tf.reshape(planned_local_env, [self.batch_size, -1]), axis=1)

        speed = np.linalg.norm(action, axis=1)

        features = tf.stack((wiggle, rope_lengths, next_rope_lengths, occ, speed), axis=1)

        accept_probability = self.output_layer(features)
        return accept_probability


def check_validation(val_tf_dataset, loss, net, from_image=False):
    val_losses = []
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    val_accuracy.reset_states()
    for val_example_dict_batch in val_tf_dataset:
        val_true_labels_batch = val_example_dict_batch['label']
        if from_image:
            val_predictions_batch = net.from_image(val_example_dict_batch['image'])
        else:
            val_predictions_batch = net(val_example_dict_batch)
        val_loss_batch = loss(y_true=val_true_labels_batch, y_pred=val_predictions_batch)
        val_accuracy.update_state(y_true=val_true_labels_batch, y_pred=val_predictions_batch)
        val_losses.append(val_loss_batch)
    val_losses = np.array(val_losses)
    return val_losses, val_accuracy


def train(hparams, train_tf_dataset, val_tf_dataset, log_path, args, from_image: bool = False):
    if from_image:
        raise ValueError("feature classifier only works on new type datasets")

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    loss = tf.keras.losses.BinaryCrossentropy()
    train_epoch_accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    batch_accuracy = tf.keras.metrics.BinaryAccuracy(name='batch_accuracy')
    net = FeatureClassifier(hparams=hparams, batch_size=args.batch_size)
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
                    train_predictions_batch = net(train_example_dict_batch)
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


def eval(hparams, test_tf_dataset, args, from_image: bool = False):
    if from_image:
        raise ValueError("feature classifier only works on new type datasets")
    net = FeatureClassifier(hparams=hparams, batch_size=args.batch_size)
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
        test_batch_predictions = net(test_example_dict)
        test_predictions.extend(test_batch_predictions.numpy().flatten().tolist())
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

    plt.hist(test_predictions, bins=10)
    plt.xlabel("accept probability")
    plt.ylabel("count")
    plt.show(block=True)

    test_loss = np.mean(test_losses)
    test_accuracy = accuracy.result().numpy() * 100
    print("Test Loss:     {:7.4f}".format(test_loss))
    print("Test Accuracy: {:5.3f}%".format(test_accuracy))

    print("|        | label 0 | label 1 |")
    print("| pred 0 | {:6d} | {:6d} |".format(tn, fn))
    print("| pred 1 | {:6d} | {:6d} |".format(fp, tp))


class FeatureClassifierWrapper(BaseClassifier):

    def __init__(self, path: pathlib.Path, batch_size: int, show: bool = False):
        super().__init__()
        model_hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.net = FeatureClassifier(hparams=self.model_hparams, batch_size=batch_size)
        self.batch_size = batch_size
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        self.ckpt.restore(self.manager.latest_checkpoint)
        self.show = show

    def predict(self, local_env_data: List, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> float:
        """
        :param local_env_data:
        :param s1: [batch, n_state] float64
        :param s2: [batch, n_state] float64
        :param action: [batch, n_action] float64
        :return: [batch, 1] float64
        """
        data_s, res_s, origin_s, extent_s = link_bot_sdf_utils.batch_occupancy_data(local_env_data)
        test_x = {
            'planned_state': tf.convert_to_tensor(s1, dtype=tf.float32),
            'planned_state_next': tf.convert_to_tensor(s2, dtype=tf.float32),
            'action': tf.reshape(tf.convert_to_tensor(action, dtype=tf.float32), [self.batch_size, -1]),
            'planned_local_env/env': tf.convert_to_tensor(data_s, dtype=tf.float32),
            'planned_local_env/origin': tf.convert_to_tensor(origin_s, dtype=tf.float32),
            'planned_local_env/extent': tf.convert_to_tensor(extent_s, dtype=tf.float32),
            'resolution': tf.convert_to_tensor(res_s, dtype=tf.float32),
        }
        # accept_probabilities = self.net(test_x)
        # accept_probabilities = accept_probabilities.numpy()

        # FIXME: debugging
        speed = np.linalg.norm(action)
        if speed > 0.20:
            accept_probabilities = np.array([[0]])
        else:
            accept_probabilities = np.array([[1]])
        #################

        accept_probabilities = accept_probabilities.astype(np.float64)[:, 0]

        if self.show:
            title = "n_parallel_calls(accept) = {:5.3f}".format(accept_probabilities.squeeze())
            plot_classifier_data(planned_env=local_env_data[0].data,
                                 planned_env_extent=local_env_data[0].extent,
                                 planned_state=s1[0],
                                 planned_next_state=s2[0],
                                 actual_env=None,
                                 actual_env_extent=None,
                                 action=action[0],
                                 state=None,
                                 next_state=None,
                                 title=title,
                                 label=None)
            plt.show()

        return accept_probabilities


model = FeatureClassifier
