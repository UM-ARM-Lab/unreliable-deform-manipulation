#!/usr/bin/env python
from __future__ import print_function

import json
import pathlib
from typing import List

import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_data.visualization import plottable_rope_configuration
from link_bot_pycommon.link_bot_sdf_utils import point_to_idx


class HandDesignedObsClassifier(BaseClassifier):

    def __init__(self, path: pathlib.Path, batch_size: int, show: bool = False):
        super().__init__()
        model_hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.batch_size = batch_size

    def predict(self, local_env_data: List, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> List[float]:
        """
        :param local_env_data:
        :param s1: [batch, n_state] float64
        :param s2: [batch, n_state] float64
        :param action: [batch, n_action] float64
        :return: [batch, 1] float64
        """
        local_env_data = local_env_data[0]
        local_env = local_env_data.data
        s2 = s2[0]

        speed = np.linalg.norm(action)

        # angles = [angle_2d(deltas[i], deltas[i + 1]) for i in range(len(deltas) - 1)]
        # wiggle = np.rad2deg(np.mean(np.abs(angles)))

        next_points = s2.reshape(-1, 2)
        next_distances = np.linalg.norm(next_points[1:] - next_points[:-1], axis=1)
        next_rope_length = np.sum(next_distances)

        occ = local_env.sum()

        if next_rope_length > self.model_hparams['max_length']:
            return [0.]
        elif next_rope_length < self.model_hparams['min_length']:
            return [0.]
        elif speed > self.model_hparams['max_speed']:
            return [0.]

        xs2, ys2 = plottable_rope_configuration(s2)

        def _check_is_free_space(xs, ys):
            is_free_space = True
            for x, y in zip(xs, ys):
                row, col = point_to_idx(x, y, local_env_data.resolution, origin=local_env_data.origin)
                try:
                    # 1 means obstacle, aka in collision
                    d = local_env[row, col]
                    point_not_in_collision = not d
                    # prediction of True means not in collision
                    is_free_space = is_free_space and point_not_in_collision
                except IndexError:
                    pass
            return is_free_space

        if not _check_is_free_space(xs2, ys2):
            return [0.0]

        occ_prob = np.exp(-occ * self.model_hparams['occ_k'])

        return [occ_prob]


model = HandDesignedObsClassifier
