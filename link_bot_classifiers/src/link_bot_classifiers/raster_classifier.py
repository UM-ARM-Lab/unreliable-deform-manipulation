#!/usr/bin/env python
from __future__ import print_function

import json
import pathlib
from typing import Dict

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore
from tensorflow import keras

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_planning.params import LocalEnvParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.link_bot_sdf_utils import OccupancyData
from moonshine.base_model import BaseModel
from moonshine.numpy_utils import add_batch
from moonshine.raster_points_layer import make_transition_image, make_traj_image


class RasterClassifier(BaseModel):

    def __init__(self, hparams: Dict, batch_size: int, *args, **kwargs):
        super().__init__(hparams, batch_size, *args, **kwargs)
        self.dynamics_dataset_hparams = self.hparams['classifier_dataset_hparams']['fwd_model_hparams'][
            'dynamics_dataset_hparams']
        self.n_action = self.dynamics_dataset_hparams['n_action']
        self.batch_size = batch_size

        self.local_env_params = LocalEnvParams.from_json(self.dynamics_dataset_hparams['local_env_params'])

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
        if self.hparams['batch_norm']:
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

    def _conv(self, image):
        # feed into a CNN
        conv_z = image
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z

        return out_conv_z

    def call(self, input_dict: dict, training=None, mask=None):
        # Choose what key to use, so depending on how the model was trained it will expect a transition_image or trajectory_image
        image = input_dict[self.hparams['image_key']]
        state = input_dict['planned_state/link_bot']
        action = input_dict['action']
        next_state = input_dict['planned_state_next/link_bot']
        out_conv_z = self._conv(image)
        conv_output = self.conv_flatten(out_conv_z)

        # FIXME: eventually remove this if I retrained fs-raster
        if 'mixed' in self.hparams and self.hparams['mixed']:
            conv_output = tf.concat((conv_output, state, action, next_state), axis=1)

        if self.hparams['batch_norm']:
            conv_output = self.batch_norm(conv_output)

        z = conv_output
        for dropout_layer, dense_layer in zip(self.dropout_layers, self.dense_layers):
            h = dropout_layer(z)
            z = dense_layer(h)
        out_h = z

        accept_probability = self.output_layer(out_h)
        return accept_probability


class RasterClassifierWrapper(BaseClassifier):

    def __init__(self, path: pathlib.Path, batch_size: int):
        super().__init__()
        model_hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.net = RasterClassifier(hparams=self.model_hparams, batch_size=batch_size)
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        self.ckpt.restore(self.manager.latest_checkpoint)

    def predict_transition(self, local_env_data: OccupancyData, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> float:
        """
        :param local_env_data:
        :param s1: [n_state] float64
        :param s2: [n_state] float64
        :param action: [n_action] float64
        :return: [1] float64
        """
        origin = local_env_data.origin
        res = local_env_data.resolution[0]
        local_env = local_env_data.data
        action_in_image = self.model_hparams['action_in_image']
        image = make_transition_image(local_env, s1, action, s2, res, origin, action_in_image)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        net_inputs = {
            'transition_image': image,
            'planned_state/link_bot': tf.expand_dims(tf.convert_to_tensor(s1, tf.float32), axis=0),
            'action': tf.expand_dims(tf.convert_to_tensor(action, tf.float32), axis=0),
            'planned_state_next/link_bot': tf.expand_dims(tf.convert_to_tensor(s2, tf.float32), axis=0),
        }

        accept_probabilities = self.net(net_inputs)
        accept_probabilities = accept_probabilities.numpy()
        accept_probabilities = accept_probabilities.astype(np.float64).squeeze()

        return accept_probabilities

    def predict_traj(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        s1 = states['link_bot'][-2]
        s2 = states['link_bot'][-1]
        action = actions[-1]

        batched_inputs = add_batch(full_env.data, full_env.origin, full_env.resolution[0], states['link_bot'])
        image = make_traj_image(*batched_inputs)
        net_inputs = {
            'trajectory_image': image,
            'planned_state/link_bot': tf.expand_dims(tf.convert_to_tensor(s1, tf.float32), axis=0),
            'action': tf.expand_dims(tf.convert_to_tensor(action, tf.float32), axis=0),
            'planned_state_next/link_bot': tf.expand_dims(tf.convert_to_tensor(s2, tf.float32), axis=0),
        }

        accept_probabilities = self.net(net_inputs)
        accept_probabilities = accept_probabilities.numpy()
        accept_probabilities = accept_probabilities.astype(np.float64).squeeze()

        return accept_probabilities

    def predict(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        # TODO: pass in dicts to predict_transition, remove specialization for link_bot key
        image_key = self.model_hparams['image_key']
        if image_key == 'transition_image':
            head_point = states['link_bot'][-2]
            local_env_params = self.net.local_env_params
            local_env, local_env_origin = link_bot_sdf_utils.get_local_env_and_origin(*add_batch(head_point,
                                                                                                 full_env.data,
                                                                                                 full_env.origin),
                                                                                      local_h_rows=local_env_params.h_rows,
                                                                                      local_w_cols=local_env_params.w_cols,
                                                                                      res=full_env.resolution[0])
            # remove batch dim with [0]
            local_env = OccupancyData(data=local_env[0],
                                      resolution=full_env.resolution,
                                      origin=local_env_origin[0])
            return self.predict_transition(local_env_data=local_env,
                                           s1=states['link_bot'][-2],
                                           s2=states['link_bot'][-1],
                                           action=actions[-1])
        elif image_key == 'trajectory_image':
            return self.predict_traj(full_env, states, actions)
        else:
            raise ValueError('invalid image_key')
