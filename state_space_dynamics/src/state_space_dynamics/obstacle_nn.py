import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore
from tensorflow import keras

from link_bot_planning.params import LocalEnvParams, FullEnvParams
from link_bot_pycommon import link_bot_sdf_utils
from moonshine.action_smear_layer import smear_action
from moonshine.numpy_utils import add_batch, dict_of_sequences_to_sequence_of_dicts
from moonshine.raster_points_layer import differentiable_raster
from moonshine.tensorflow_train_test_loop import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class ObstacleNN(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int):
        super().__init__(hparams, batch_size)
        self.initial_epoch = 0

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
        local_env = link_bot_sdf_utils.differentiable_get_local_env(head_point_t,
                                                                    padded_full_envs,
                                                                    full_env_origin,
                                                                    padding,
                                                                    self.local_env_params.h_rows,
                                                                    self.local_env_params.w_cols,
                                                                    self.local_env_params.res)
        env_h_rows = tf.convert_to_tensor(self.local_env_params.h_row, tf.int64)
        env_w_cols = tf.convert_to_tensor(self.local_env_params.w_cols, tf.int64)
        return local_env, local_env_origin, env_h_rows, env_w_cols

    def call(self, dataset_element, training=None, mask=None):
        input_dict, _ = dataset_element
        actions = input_dict['action']
        input_sequence_length = actions.shape[1]

        substates_0 = []
        for state_key, n in self.used_states_description.items():
            substate_0 = input_dict[state_key][:, 0]
            substates_0.append(substate_0)

        s_0 = tf.concat(substates_0, axis=1)

        resolution = input_dict['full_env/res']
        full_env = input_dict['full_env/env']
        full_env_origin = input_dict['full_env/origin']

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]

            action_t = actions[:, t]

            head_point_t = s_t[:, -2:]
            if 'use_full_env' in self.hparams and self.hparams['use_full_env']:
                env = full_env
                env_origin = full_env_origin
                env_h_rows = tf.convert_to_tensor(self.full_env_params.h_rows, tf.int64)
                env_w_cols = tf.convert_to_tensor(self.full_env_params.w_cols, tf.int64)
            else:
                # the local environment used at each time step is centered on the current point of the head
                env, env_origin, env_h_rows, env_w_cols = self.get_local_env(head_point_t, full_env_origin, full_env)

            rope_image_t = differentiable_raster(s_t, resolution, env_origin, env_h_rows, env_w_cols)

            # FIXME: this differentiable already, but we need to implement it in TensorFlow
            action_image_t = tf.numpy_function(smear_action, [action_t, env_h_rows, env_w_cols], tf.float32)

            env = tf.expand_dims(env, axis=3)

            # CNN
            z_t = self.concat([rope_image_t, env, action_image_t])
            for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
                z_t = conv_layer(z_t)
                z_t = pool_layer(z_t)
            conv_z_t = self.flatten_conv_output(z_t)
            if self.hparams['mixed']:
                full_z_t = self.concat2([s_t, action_t, conv_z_t])
            else:
                full_z_t = conv_z_t

            # dense layers
            for dense_layer in self.dense_layers:
                full_z_t = dense_layer(full_z_t)

            if self.hparams['residual']:
                # residual prediction, otherwise just take the final hidden representation as the next state
                residual_t = full_z_t
                s_t_plus_1_flat = s_t + residual_t
            else:
                s_t_plus_1_flat = full_z_t

            pred_states.append(s_t_plus_1_flat)

        pred_states = tf.stack(pred_states, axis=1)

        # Split the big state vectors up by state name/dim
        start_idx = 0
        output_states_dict = {}
        for name, n in self.used_states_description.items():
            end_idx = start_idx + n
            output_states_dict[state_key] = pred_states[:, :, start_idx:end_idx]
            start_idx += n

        return output_states_dict


class ObstacleNNWrapper(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path, batch_size: int):
        super().__init__(model_dir, batch_size)
        self.net = ObstacleNN(hparams=self.hparams, batch_size=batch_size)
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_dir, max_to_keep=1)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)

    def propagate_differentiable(self,
                                 full_env: np.ndarray,
                                 full_env_origin: np.ndarray,
                                 res: float,
                                 start_states: Dict[str, np.ndarray],
                                 actions: tf.Variable) -> List[Dict]:
        """
        :param full_env:        (H, W)
        :param full_env_origin: (2)
        :param res:             scalar
        :param start_states:          each value in the dictionary should be of shape (batch, n_state)
        :param actions:        (T, 2)
        :return: states:       each value in the dictionary should be a of shape [batch, T+1, n_state)
        """
        test_x = {
            # shape: T, 2
            'action': tf.convert_to_tensor(actions, dtype=tf.float32),
            # shape: 1
            'res': tf.convert_to_tensor(res, dtype=tf.float32),
            # shape: H, W
            'full_env/env': tf.convert_to_tensor(full_env, dtype=tf.float32),
            # shape: 2
            'full_env/origin': tf.convert_to_tensor(full_env_origin, dtype=tf.float32),
        }

        for state_key, v in start_states.items():
            # handles conversion from double -> float
            state = tf.convert_to_tensor(v, dtype=tf.float32)
            first_state = tf.reshape(state, state.shape[0])
            test_x[state_key] = first_state

        test_x = add_batch(test_x)
        predictions = self.net((test_x, None))
        predictions = dict_of_sequences_to_sequence_of_dicts(predictions)

        return predictions


model = ObstacleNN
