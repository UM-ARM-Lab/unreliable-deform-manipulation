#!/usr/bin/env python
from __future__ import print_function

import json
import pathlib
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from colorama import Fore

from link_bot_classifiers.base_classifier import BaseClassifier
from moonshine.base_model import BaseModel


class FeatureClassifier(BaseModel):

    def __init__(self, hparams, batch_size, *args, **kwargs):
        super().__init__(hparams, batch_size, *args, **kwargs)

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

    def compute_features(self, input_dict: Dict):
        planned_state = input_dict['planned_state/link_bot']
        action = input_dict['action']
        planned_next_state = input_dict['planned_state_next/link_bot']

        points = tf.reshape(planned_state, [self.batch_size, -1, 2])
        next_points = tf.reshape(planned_next_state, [self.batch_size, -1, 2])
        distances = tf.norm(points, axis=2)
        next_distances = tf.norm(next_points, axis=2)

        features = []
        if 'distances' in self.hparams['features']:
            features.extend([distances, next_distances])
        if 'states' in self.hparams['features']:
            features.extend([planned_state, planned_next_state])
        if 'action' in self.hparams['features']:
            features.append(action)

        features = tf.concat(features, axis=1)
        return features

    def call(self, input_dict: dict, training=None, mask=None):
        features = self.compute_features(input_dict)

        z = features
        for dropout_layer, dense_layer in zip(self.dropout_layers, self.dense_layers):
            h = dropout_layer(z)
            z = dense_layer(h)
        out_h = z

        accept_probability = self.output_layer(out_h)
        return accept_probability


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
        test_x = {
            'planned_state/link_bot': tf.convert_to_tensor(s1, dtype=tf.float32),
            'planned_state_next/link_bot': tf.convert_to_tensor(s2, dtype=tf.float32),
        }
        accept_probabilities = self.net(test_x)
        accept_probabilities = accept_probabilities.numpy()
        accept_probabilities = accept_probabilities.astype(np.float64)[:, 0]

        return accept_probabilities


model = FeatureClassifier
