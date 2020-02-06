#!/usr/bin/env python
from __future__ import print_function

import pathlib
from typing import List

import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_data.image_classifier_dataset import ImageClassifierDataset
from link_bot_data.new_classifier_dataset import NewClassifierDataset
from link_bot_pycommon import link_bot_sdf_utils


class NearestNeighborClassifier(BaseClassifier):

    def __init__(self, classifier_dataset_dirs: List[pathlib.Path], dataset_type: str):
        super().__init__()
        if dataset_type == 'image':
            classifier_dataset = ImageClassifierDataset(classifier_dataset_dirs)
            self.dataset = classifier_dataset.get_datasets(mode='train', shuffle=False, batch_size=None, balance_key=None)
        elif dataset_type == 'new':
            classifier_dataset = NewClassifierDataset(classifier_dataset_dirs)
            self.dataset = classifier_dataset.get_datasets(mode='train', shuffle=False, batch_size=None, balance_key=None)


    def predict(self, local_env_data: List, s1_s: np.ndarray, s2_s: np.ndarray) -> float:
        """
        :param local_env_datas:
        :param s1: [batch, 6] float64
        :param s2: [batch, 6] float64
        :return: [batch, 1] float6n
        """
        # lookup nearest neighbor in dataset

        # How to do the ensemble here without training the free-space model to actually take into account the local env?

        data_s, res_s, origin_s, extent_s = link_bot_sdf_utils.batch_occupancy_data(local_env_data)
        test_x = {
            'planned_state': tf.convert_to_tensor(s1_s, dtype=tf.float32),
            'planned_state_next': tf.convert_to_tensor(s2_s, dtype=tf.float32),
            'planned_local_env/env': tf.convert_to_tensor(data_s, dtype=tf.float32),
            'planned_local_env/origin': tf.convert_to_tensor(origin_s, dtype=tf.float32),
            'planned_local_env/extent': tf.convert_to_tensor(extent_s, dtype=tf.float32),
            'resolution': tf.convert_to_tensor(res_s, dtype=tf.float32),
        }
        accept_probabilities = self.net(test_x)[-1]
        accept_probabilities = accept_probabilities.numpy()
        accept_probabilities = accept_probabilities.astype(np.float64)[:, 0]

        return accept_probabilities


# TODO: put this in for all classifier and dynamics models
model = NearestNeighborClassifier
