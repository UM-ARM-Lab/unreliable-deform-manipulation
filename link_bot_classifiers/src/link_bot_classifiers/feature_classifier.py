#!/usr/bin/env python
from __future__ import print_function

import json
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_pycommon import link_bot_sdf_utils
from moonshine.base_model import BaseModel


class FeatureClassifier(BaseModel):

    def __init__(self, hparams, batch_size, *args, **kwargs):
        super().__init__(hparams, batch_size, *args, **kwargs)

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def compute_features(self, input_dict: Dict):
        planned_state = input_dict['planned_state/link_bot']
        planned_next_state = input_dict['planned_state_next/link_bot']

        points = tf.reshape(planned_state, [self.batch_size, -1, 2])
        distances = tf.norm(points, axis=2)
        next_points = tf.reshape(planned_next_state, [self.batch_size, -1, 2])
        next_distances = tf.norm(next_points, axis=2)

        features = tf.concat((distances, next_distances), axis=1)
        return features

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
        features = self.compute_features(input_dict)

        accept_probability = self.output_layer(features)
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
