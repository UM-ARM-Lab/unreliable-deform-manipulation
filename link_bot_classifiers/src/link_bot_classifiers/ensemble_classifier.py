#!/usr/bin/env python
from __future__ import print_function

import json
import pathlib
from typing import List

import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_planning import model_utils
from link_bot_pycommon import link_bot_sdf_utils


class EnsembleClassifier(BaseClassifier):

    def __init__(self, ensembles_path: pathlib.Path, show: bool = False):
        super().__init__()

        # load all the models in the ensemble
        self.models = []
        for model_dir in ensembles_path.iterdir():
            # TODO: look up the model_type from the hparams in model_dir
            model_type = 'nn'
            model, _ = model_utils.load_generic_model(model_dir, model_type)

            model_hparams_file = model_dir / 'hparams.json'
            model_hparams = json.load(model_hparams_file.open('r'))
            net = model(hparams=model_hparams)
            ckpt = tf.train.Checkpoint(net=net)
            manager = tf.train.CheckpointManager(ckpt, model_dir)
            if manager.latest_checkpoint:
                print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
            ckpt.restore(manager.latest_checkpoint)
            self.models.append(net)

    def predict(self, local_env_data: List, s1_s: np.ndarray, s2_s: np.ndarray) -> float:
        """
        :param local_env_datas:
        :param s1: [batch, 6] float64
        :param s2: [batch, 6] float64
        :return: [batch, 1] float6n
        """

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
model = EnsembleClassifier
