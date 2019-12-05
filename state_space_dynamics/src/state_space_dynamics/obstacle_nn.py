import json
import pathlib
import time
from typing import List

import numpy as np
import progressbar
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore, Style

from link_bot_planning.params import LocalEnvParams
from link_bot_pycommon import experiments_util, link_bot_pycommon
from moonshine.action_smear_layer import action_smear_layer
from moonshine.raster_points_layer import RasterPoints
from state_space_dynamics.base_forward_model import BaseForwardModel


class ObstacleNN(tf.keras.Model):

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_epoch = 0
        self.hparams = tf.contrib.checkpoint.NoDependency(hparams)

        local_env_params = LocalEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['local_env_params'])
        self.raster = RasterPoints([local_env_params.h_rows, local_env_params.w_cols])
        self.concat = layers.Concatenate()
        self.concat2 = layers.Concatenate()
        self.action_smear = action_smear_layer(1,
                                               self.hparams['dynamics_dataset_hparams']['n_action'],
                                               local_env_params.h_rows,
                                               local_env_params.w_cols)
        self.dense_layers = []
        for fc_layer_size in self.hparams['fc_layer_sizes']:
            self.dense_layers.append(layers.Dense(fc_layer_size, activation='relu', use_bias=True))
        self.dense_layers.append(layers.Dense(self.hparams['dynamics_dataset_hparams']['n_state'], activation=None))
        self.conv_layers = []
        for n_filters, filter_size in self.hparams['conv_filters']:
            self.conv_layers.append((layers.Conv2D(n_filters, filter_size, activation='relu', use_bias=True),
                                     layers.MaxPool2D(2)))
        self.flatten_conv_output = layers.Flatten()

    def call(self, input_dict, training=None, mask=None):
        actions = input_dict['action_s']
        input_sequence_length = actions.shape[1]
        s_0 = tf.expand_dims(input_dict['state_s'][:, 0], axis=2)
        local_env = tf.expand_dims(input_dict['actual_local_env_s/env'][:, 0], axis=3)
        resolution = input_dict['resolution_s'][:, 0]
        res_2d = tf.expand_dims(tf.tile(resolution, [1, 2]), axis=1)
        origin = tf.expand_dims(input_dict['actual_local_env_s/origin'][:, 0], axis=1)

        gen_states = [s_0]
        for t in range(input_sequence_length):
            s_t = gen_states[-1]
            s_t_squeeze = tf.squeeze(s_t, squeeze_dims=2)
            action_t = actions[:, t]

            # filters out out of bounds points internally with no warnings
            rope_image_s = self.raster([tf.expand_dims(s_t, axis=1), res_2d, origin])
            rope_image_t = rope_image_s[:, 0]
            action_image_s = self.action_smear(action_t)
            action_image_t = action_image_s[:, 0]

            # CNN
            z_t = self.concat([rope_image_t, local_env, action_image_t])
            for conv_layer, pool_layer in self.conv_layers:
                z_t = conv_layer(z_t)
                z_t = pool_layer(z_t)
            conv_z_t = self.flatten_conv_output(z_t)
            if self.hparams['mixed']:
                full_z_t = self.concat2([s_t_squeeze, action_t, conv_z_t])
            else:
                full_z_t = conv_z_t

            # dense layers
            for dense_layer in self.dense_layers:
                full_z_t = dense_layer(full_z_t)

            # residual prediction, otherwise just take the final hidden representation as the next state
            if self.hparams['residual']:
                ds_t = tf.expand_dims(full_z_t, axis=2)
                s_t_plus_1_flat = s_t + ds_t
            else:
                s_t_plus_1_flat = tf.expand_dims(full_z_t, axis=2)

            gen_states.append(s_t_plus_1_flat)

        gen_states = tf.stack(gen_states)
        gen_states = tf.transpose(gen_states, [1, 0, 2, 3])
        gen_states = tf.squeeze(gen_states, squeeze_dims=3)
        return gen_states


def eval(hparams, test_tf_dataset, args):
    net = ObstacleNN(hparams=hparams)
    ckpt = tf.train.Checkpoint(net=net)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)

    # TODO: should we make sure that we use the same loss upon restoring the checkpoint?
    loss = tf.keras.losses.MeanSquaredError()

    test_losses = []
    test_position_errors = []
    for test_x, test_y in test_tf_dataset:
        test_true_states = test_y['output_states']
        test_gen_states = net(test_x)
        batch_test_loss = loss(y_true=test_true_states, y_pred=test_gen_states)
        test_gen_points = tf.reshape(test_gen_states, [test_gen_states.shape[0], test_gen_states.shape[1], 3, 2])
        test_true_points = tf.reshape(test_true_states, [test_true_states.shape[0], test_true_states.shape[1], 3, 2])
        batch_test_position_error = tf.reduce_mean(tf.linalg.norm(test_gen_points - test_true_points, axis=3), axis=0)
        test_losses.append(batch_test_loss)
        test_position_errors.append(batch_test_position_error)
    test_loss = np.mean(test_losses)
    test_position_error = np.mean(test_position_errors)
    print("Test Loss:  {:8.5f}".format(test_loss))
    print("Test Error: " + Style.BRIGHT + "{:8.4f}(m)".format(test_position_error) + Style.RESET_ALL)


def eval_angled(net, test_tf_dataset):
    angles = []
    errors = []
    for test_x, test_y in test_tf_dataset:
        test_true_states = test_y['output_states']
        test_gen_states = net(test_x)
        for true_state_seq, gen_state_seq in zip(test_true_states, test_gen_states):
            true_initial_state = true_state_seq[0]
            true_final_state = true_state_seq[-1]
            gen_final_state = gen_state_seq[-1]
            angle = link_bot_pycommon.angle_from_configuration(true_initial_state)
            error = np.linalg.norm(true_final_state - gen_final_state)
            angles.append(angle)
            errors.append(error)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(angles, errors)
    plt.plot([0, np.pi], [0, 0], c='k')
    plt.xlabel("angle (rad)")
    plt.ylabel("increase in prediction error in R6 (m)")
    plt.show()


def train(hparams, train_tf_dataset, val_tf_dataset, log_path, args):
    optimizer = tf.train.AdamOptimizer()
    loss = tf.keras.losses.MeanSquaredError()
    net = ObstacleNN(hparams=hparams)
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
        with open(hparams_path, 'w') as hparams_file:
            hparams['log path'] = str(full_log_path)
            hparams['datasets'] = str(args.dataset_dirs)
            hparams_file.write(json.dumps(hparams, indent=2))

        writer = tf.contrib.summary.create_file_writer(logdir=full_log_path)

    def train_loop():
        ################
        # test the loss before any training occurs
        ################
        val_losses = []
        for val_x, val_y in progressbar.progressbar(val_tf_dataset):
            true_val_states = val_y['output_states']
            val_gen_states = net(val_x)
            batch_val_loss = loss(y_true=true_val_states, y_pred=val_gen_states)
            val_losses.append(batch_val_loss)
        val_loss = np.mean(val_losses)
        print("Validation loss before any training: " + Style.BRIGHT + "{:8.5f}".format(val_loss) + Style.RESET_ALL)

        for epoch in range(args.epochs):
            ################
            # train
            ################
            # metrics are averaged across batches in the epoch
            batch_losses = []
            epoch_t0 = time.time()
            train_batch_x = None
            train_batch_y = None
            for train_batch_x, train_batch_y in progressbar.progressbar(train_tf_dataset):
                batch_t0 = time.time()
                true_train_states = train_batch_y['output_states']
                with tf.GradientTape() as tape:
                    pred_states = net(train_batch_x)
                    training_batch_loss = loss(y_true=true_train_states, y_pred=pred_states)
                variables = net.trainable_variables
                gradients = tape.gradient(training_batch_loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                batch_losses.append(training_batch_loss.numpy())

                global_step.assign_add(1)

                if args.log:
                    for grad, var in zip(gradients, variables):
                        tf.contrib.summary.histogram(var.name + '_grad', grad)
                dt_per_step = time.time() - batch_t0
                if args.verbose >= 3:
                    print("{:4.1f}ms/step".format(dt_per_step * 1000.0))
            dt_per_epoch = time.time() - epoch_t0

            training_loss = np.mean(batch_losses)
            print("Epoch: {:5d}, Time {:4.1f}s, Training loss: {:8.5f}".format(epoch, dt_per_epoch, training_loss))
            if args.log:
                tf.contrib.summary.scalar("training loss", training_loss)

            ################
            # validation
            ################
            if epoch % args.validation_every == 0:
                val_losses = []
                for val_x, val_y in progressbar.progressbar(val_tf_dataset):
                    true_val_states = val_y['output_states']
                    val_gen_states = net(val_x)
                    batch_val_loss = loss(y_true=true_val_states, y_pred=val_gen_states)
                    val_losses.append(batch_val_loss)
                val_loss = np.mean(val_losses)
                tf.contrib.summary.scalar('validation loss', val_loss, step=int(ckpt.step))
                print("\t\t\tValidation loss: " + Style.BRIGHT + "{:8.5f}".format(val_loss) + Style.RESET_ALL)

            ################
            # Checkpoint
            ################
            if args.log and epoch % args.save_freq == 0:
                save_path = manager.save()
                print(Fore.CYAN + "Step {:6d}: Saved checkpoint {}".format(int(ckpt.step), save_path) + Fore.RESET)

        save_path = manager.save()
        print(Fore.CYAN + "Step {:6d}: Saved final checkpoint {}".format(int(ckpt.step), save_path) + Fore.RESET)

    if args.log:
        with writer.as_default(), tf.contrib.summary.always_record_summaries():
            train_loop()
    else:
        train_loop()


class ObstacleNNWrapper(BaseForwardModel):

    def __init__(self, model_dir: pathlib.Path):
        super().__init__(model_dir)
        self.net = ObstacleNN(hparams=self.hparams)
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_dir, max_to_keep=1)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)

    def predict(self, local_env_data_s: List, first_states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        batch, T, _ = actions.shape
        states = tf.convert_to_tensor(first_states, dtype=tf.float32)
        states = tf.reshape(states, [states.shape[0], 1, states.shape[1]])
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        local_env_np_s = np.array([env.data for env in local_env_data_s], dtype=np.float32)
        local_env_resolution_s = np.expand_dims(np.array([env.resolution[0:1] for env in local_env_data_s], dtype=np.float32),
                                                axis=1)
        local_env_origin_s = np.expand_dims(np.array([env.origin for env in local_env_data_s], dtype=np.float32), axis=1)
        local_env_s = tf.convert_to_tensor(local_env_np_s, dtype=tf.float32)

        test_x = {
            # must be batch, T, w, h
            'actual_local_env_s/env': local_env_s,
            # must be batch, T, 6
            'state_s': states,
            # must be batch, T, 2
            'action_s': actions,
            # must be batch, T, 1
            'resolution_s': local_env_resolution_s,
            # must be batch, T, 2
            'actual_local_env_s/origin': local_env_origin_s,
        }
        predictions = self.net(test_x)
        predicted_points = predictions.numpy().reshape([batch, T + 1, 3, 2])
        # OMPL requires "doubles", which are float64, although our network outputs float32.
        predicted_points = predicted_points.astype(np.float64)
        return predicted_points