import json
import pathlib
import time
from typing import Dict

import numpy as np
import progressbar
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore, Style
from tensorflow import keras

from link_bot_planning.params import LocalEnvParams, FullEnvParams
from link_bot_pycommon import link_bot_sdf_utils
from moonshine import experiments_util, loss_on_dicts
from moonshine.action_smear_layer import smear_action
from moonshine.numpy_utils import add_batch_to_dict
from moonshine.raster_points_layer import raster
from state_space_dynamics.base_forward_model import BaseForwardModel


class ObstacleNN(tf.keras.Model):

    def __init__(self, hparams: Dict, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_epoch = 0
        self.hparams = tf.contrib.checkpoint.NoDependency(hparams)

        self.local_env_params = LocalEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['local_env_params'])
        self.full_env_params = FullEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['full_env_params'])
        self.batch_size = batch_size
        self.concat = layers.Concatenate()
        self.concat2 = layers.Concatenate()
        # State keys is all the things we want the model to take in/predict
        self.states_keys = self.hparams['states_keys']
        self.used_states_description = {}
        self.out_dim = 0
        # the states_description lists what's available in the dataset
        for available_state_name, n in self.hparams['dynamics_dataset_hparams']['states_description'].items():
            if available_state_name in self.states_keys:
                self.used_states_description[available_state_name] = n
                self.out_dim += n
        self.dense_layers = []
        for fc_layer_size in self.hparams['fc_layer_sizes']:
            self.dense_layers.append(layers.Dense(fc_layer_size, activation='relu', use_bias=True))
        self.dense_layers.append(layers.Dense(self.out_dim, activation=None))

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

        self.flatten_conv_output = layers.Flatten()

    def get_local_env(self, head_point_t, full_env_origin, full_env):
        # NOTE: there are some really high velocities in my dataset, sort of accidentally, and the model likes to predict crazy
        #  things in this case. Since this will go out of bounds of our environment, we want to assume 0 (free space), so we pad.
        padding = 200
        paddings = [[0, 0], [padding, padding], [padding, padding]]
        padded_full_envs = tf.pad(full_env, paddings=paddings)

        local_env_origin = tf.numpy_function(link_bot_sdf_utils.get_local_env_origins,
                                             [head_point_t,
                                              full_env_origin,
                                              self.local_env_params.h_rows,
                                              self.local_env_params.w_cols,
                                              self.local_env_params.res],
                                             tf.float32)
        local_env = tf.numpy_function(link_bot_sdf_utils.get_local_env_at_in,
                                      [head_point_t,
                                       padded_full_envs,
                                       full_env_origin,
                                       padding,
                                       self.local_env_params.h_rows,
                                       self.local_env_params.w_cols,
                                       self.local_env_params.res],
                                      tf.float32)
        env_h_rows = tf.convert_to_tensor(self.local_env_params.h_row, tf.int64)
        env_w_cols = tf.convert_to_tensor(self.local_env_params.w_cols, tf.int64)
        return local_env, local_env_origin, env_h_rows, env_w_cols

    def call(self, input_dict, training=None, mask=None):
        actions = input_dict['action']
        input_sequence_length = actions.shape[1]

        substates_0 = []
        for name, n in self.used_states_description.items():
            state_key = 'state/{}'.format(name)
            substate_0 = tf.expand_dims(input_dict[state_key][:, 0], axis=2)
            substates_0.append(substate_0)

        s_0 = tf.concat(substates_0, axis=1)

        # TODO: don't collect resolution for every time step, it doesn't change
        resolution = input_dict['res'][:, 0]
        full_env = input_dict['full_env/env']
        full_env_origin = input_dict['full_env/origin']

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]
            s_t_squeeze = tf.squeeze(s_t, squeeze_dims=2)

            action_t = actions[:, t]

            head_point_t = s_t_squeeze[:, -2:]
            if 'use_full_env' in self.hparams and self.hparams['use_full_env']:
                env = full_env
                env_origin = full_env_origin
                env_h_rows = tf.convert_to_tensor(self.full_env_params.h_rows, tf.int64)
                env_w_cols = tf.convert_to_tensor(self.full_env_params.w_cols, tf.int64)
            else:
                # the local environment used at each time step is centered on the current point of the head
                env, env_origin, env_h_rows, env_w_cols = self.get_local_env(head_point_t, full_env_origin, full_env)

            rope_image_t = tf.numpy_function(raster, [s_t_squeeze, resolution, env_origin, env_h_rows, env_w_cols], tf.float32)

            action_image_t = tf.numpy_function(smear_action, [action_t, env_h_rows, env_w_cols], tf.float32)

            env = tf.expand_dims(env, axis=3)

            # CNN
            z_t = self.concat([rope_image_t, env, action_image_t])
            for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
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

            if self.hparams['residual']:
                # residual prediction, otherwise just take the final hidden representation as the next state
                residual_t = tf.expand_dims(full_z_t, axis=2)
                s_t_plus_1_flat = s_t + residual_t
            else:
                s_t_plus_1_flat = tf.expand_dims(full_z_t, axis=2)

            pred_states.append(s_t_plus_1_flat)

        pred_states = tf.stack(pred_states)
        pred_states = tf.transpose(pred_states, [1, 0, 2, 3])
        pred_states = tf.squeeze(pred_states, squeeze_dims=3)

        # Split the big state vectors up by state name/dim
        start_idx = 0
        output_states_dict = {}
        for name, n in self.used_states_description.items():
            end_idx = start_idx + n
            output_states_dict[name] = pred_states[:, :, start_idx:end_idx]
            start_idx += n

        return output_states_dict


def eval(hparams, test_tf_dataset, args):
    net = ObstacleNN(hparams=hparams, batch_size=args.batch_size)
    ckpt = tf.train.Checkpoint(net=net)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)

    loss = tf.keras.losses.MeanSquaredError()

    test_losses = []
    test_position_errors = []
    for test_x, test_y in test_tf_dataset:
        test_true_states_dict = test_y
        test_pred_states_dict = net(test_x)
        batch_test_loss = loss_on_dicts(loss, test_true_states_dict, test_pred_states_dict).numpy()
        test_true_link_bot_states = test_true_states_dict['link_bot']
        test_pred_link_bot_states = test_pred_states_dict['link_bot']
        points_shape = [test_true_link_bot_states.shape[0], test_true_link_bot_states.shape[1], -1, 2]
        test_pred_points = tf.reshape(test_pred_link_bot_states, points_shape)
        test_true_points = tf.reshape(test_true_link_bot_states, points_shape)
        batch_test_position_error = tf.reduce_mean(tf.linalg.norm(test_pred_points - test_true_points, axis=3), axis=0)
        test_losses.append(batch_test_loss)
        test_position_errors.append(batch_test_position_error)
    test_loss = np.mean(test_losses)
    test_position_error = np.mean(test_position_errors)
    print("Test Loss:  {:8.5f}".format(test_loss))
    print("Test Error: " + Style.BRIGHT + "{:8.4f}(m)".format(test_position_error) + Style.RESET_ALL)


def train(hparams, train_tf_dataset, val_tf_dataset, log_path, args, seed: int):
    optimizer = tf.train.AdamOptimizer()
    loss = tf.keras.losses.MeanSquaredError()
    net = ObstacleNN(hparams=hparams, batch_size=args.batch_size)
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
            hparams['seed'] = seed
            hparams['datasets'] = [str(d) for d in args.dataset_dirs]
            hparams_file.write(json.dumps(hparams, indent=2))

        writer = tf.contrib.summary.create_file_writer(logdir=full_log_path)

    def train_loop():
        ################
        # test the loss before any training occurs
        ################
        val_losses = []
        for val_x, val_y in progressbar.progressbar(val_tf_dataset):
            true_val_states_dict = val_y
            val_pred_states_dict = net(val_x)
            batch_val_loss = loss_on_dicts(loss, true_val_states_dict, val_pred_states_dict).numpy()
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
            for train_batch_x, train_batch_y in progressbar.progressbar(train_tf_dataset):
                batch_t0 = time.time()
                true_train_states_dict = train_batch_y
                with tf.GradientTape() as tape:
                    pred_states_dict = net(train_batch_x)
                    training_batch_loss = loss_on_dicts(loss, true_train_states_dict, pred_states_dict)
                variables = net.trainable_variables
                gradients = tape.gradient(training_batch_loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                batch_losses.append(training_batch_loss)

                global_step.assign_add(1)

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
                    true_val_states_dict = val_y
                    val_pred_states_dict = net(val_x)
                    batch_val_loss = loss_on_dicts(loss, true_val_states_dict, val_pred_states_dict).numpy()
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

    def __init__(self, model_dir: pathlib.Path, batch_size: int):
        super().__init__(model_dir)
        self.net = ObstacleNN(hparams=self.hparams, batch_size=batch_size)
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_dir, max_to_keep=1)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)

    def predict(self,
                full_env: np.ndarray,
                full_env_origin: np.ndarray,
                res: np.ndarray,
                states: Dict[str, np.ndarray],
                actions: np.ndarray) -> np.ndarray:
        """
        :param full_env:        (H, W)
        :param full_env_origin: (2)
        :param res:             scalar
        :param states:          each value in the dictionary should be of shape (batch, n_state)
        :param actions:        (T, 2)
        :return: states:       each value in the dictionary should be a of shape [batch, T+1, n_state)
        """
        T = actions.shape[0]

        test_x = {
            # must be T, 2
            'action': tf.convert_to_tensor(actions, dtype=tf.float32),
            # must be T, 1
            'res': tf.convert_to_tensor(res, dtype=tf.float32),
            # must be H, W
            'full_env/env': tf.convert_to_tensor(full_env, dtype=tf.float32),
            # must be 2
            'full_env/origin': tf.convert_to_tensor(full_env_origin, dtype=tf.float32),
            # batch dim is added below
        }

        for k, v in states.items():
            state_key = 'state/{}'.format(k)
            # handles conversion from double -> float
            state = tf.convert_to_tensor(v, dtype=tf.float32)
            # TODO: remove extra dimension of size 1?
            state = tf.reshape(state, [1, state.shape[0]])
            test_x[state_key] = state

        test_x = add_batch_to_dict(test_x)

        predictions = self.net(test_x)
        for k, v in predictions.items():
            predictions[k] = np.reshape(v.numpy(), [T + 1, -1]).astype(np.float64)

        return predictions


model = ObstacleNN
