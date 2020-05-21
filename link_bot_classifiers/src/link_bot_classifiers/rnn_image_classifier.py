#!/usr/bin/env python
import json
import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore
from tensorflow import keras

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_data.link_bot_dataset_utils import add_planned, NULL_PAD_VALUE
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_pycommon import make_dict_float32
from link_bot_pycommon.params import FullEnvParams
from moonshine.action_smear_layer import smear_action_differentiable
from moonshine.get_local_environment import get_local_env_and_origin_differentiable as get_local_env
from moonshine.image_functions import raster_differentiable
from moonshine.moonshine_utils import add_batch, dict_of_numpy_arrays_to_dict_of_tensors, flatten_batch_and_time
from moonshine.tensorflow_train_test_loop import MyKerasModel


class RNNImageClassifier(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams, batch_size, scenario=scenario)

        self.classifier_dataset_hparams = self.hparams['classifier_dataset_hparams']
        self.dynamics_dataset_hparams = self.classifier_dataset_hparams['fwd_model_hparams']['dynamics_dataset_hparams']
        self.n_action = self.dynamics_dataset_hparams['n_action']
        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.rope_image_k = self.hparams['rope_image_k']

        self.states_keys = self.hparams['states_keys']

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

        self.mask = layers.Masking(mask_value=NULL_PAD_VALUE)
        self.lstm = layers.LSTM(self.hparams['rnn_size'], unroll=True, return_sequences=True)
        self.output_layer = layers.Dense(1, activation=None)
        self.sigmoid = layers.Activation("sigmoid")

    @tf.function
    def make_traj_images(self,
                         environment,
                         states_dict_batch_time,
                         local_env_center_point_batch_time,
                         padded_actions,
                         time):
        """
        :return: [batch, time, h, w, 1 + n_points]
        """
        actions_batch_time = tf.reshape(padded_actions, [-1] + padded_actions.shape.as_list()[2:])
        batch_and_time = self.batch_size * time
        env_batch_time = tf.tile(environment['full_env/env'], [time, 1, 1])
        env_origin_batch_time = tf.tile(environment['full_env/origin'], [time, 1])
        env_res_batch_time = tf.tile(environment['full_env/res'], [time])

        # this will produce images even for "null" data,
        # but are masked out in the RNN, and not actually used in the computation
        local_env_batch_time, local_env_origin_batch_time = get_local_env(center_point=local_env_center_point_batch_time,
                                                                          full_env=env_batch_time,
                                                                          full_env_origin=env_origin_batch_time,
                                                                          res=env_res_batch_time,
                                                                          local_h_rows=self.local_env_h_rows,
                                                                          local_w_cols=self.local_env_w_cols)

        concat_args = []
        for planned_state in states_dict_batch_time.values():
            planned_rope_image = raster_differentiable(state=planned_state,
                                                       res=env_res_batch_time,
                                                       origin=local_env_origin_batch_time,
                                                       h=self.local_env_h_rows,
                                                       w=self.local_env_w_cols,
                                                       k=self.rope_image_k,
                                                       batch_size=batch_and_time)
            concat_args.append(planned_rope_image)

        if self.hparams['action_in_image']:
            action_image = smear_action_differentiable(actions_batch_time, self.local_env_h_rows, self.local_env_w_cols)
            concat_args.append(action_image)

        concat_args.append(tf.expand_dims(local_env_batch_time, axis=3))
        images_batch_time = tf.concat(concat_args, axis=3)
        return images_batch_time

    @tf.function
    def _conv(self, images, time):
        # merge batch & time dimensions
        conv_z = images
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z
        out_conv_z = tf.reshape(out_conv_z, [time, self.batch_size, -1])
        # un-merge batch & time dimensions

        return out_conv_z

    @tf.function
    def call(self, input_dict: Dict, training, **kwargs):
        # First flatten batch & time
        transposed_states_dict = {k: tf.transpose(input_dict[add_planned(k)], [1, 0, 2]) for k in self.states_keys}
        states_dict_batch_time = flatten_batch_and_time(transposed_states_dict)
        padded_action = tf.pad(input_dict['action'], [[0, 0], [0, 1], [0, 0]])
        time = int(padded_action.shape[1])
        local_env_center_point_batch_time = self.scenario.local_environment_center_differentiable(states_dict_batch_time)
        images_batch_and_time = self.make_traj_images(environment=self.scenario.get_environment_from_example(input_dict),
                                                      states_dict_batch_time=states_dict_batch_time,
                                                      local_env_center_point_batch_time=local_env_center_point_batch_time,
                                                      padded_actions=padded_action,
                                                      time=time)

        conv_output = self._conv(images_batch_and_time, time)
        conv_output = tf.transpose(conv_output, [1, 0, 2])  # undo transpose

        concat_args = [conv_output, padded_action]
        for state_key in self.states_keys:
            planned_state_key = add_planned(state_key)
            state = input_dict[planned_state_key]
            if 'use_local_frame' in self.hparams and self.hparams['use_local_frame']:
                # note this assumes all state vectors are[x1,y1,...,xn,yn]
                time = state.shape[1]
                points = tf.reshape(state, [self.batch_size, time, -1, 2])
                points = points - points[:, :, tf.newaxis, 0]
                state = tf.reshape(points, [self.batch_size, time, -1])
            concat_args.append(state)

        if self.hparams['stdev']:
            stdevs = input_dict[add_planned('stdev')]
            concat_args.append(stdevs)

        conv_output = tf.concat(concat_args, axis=2)

        if self.hparams['batch_norm']:
            conv_output = self.batch_norm(conv_output, training=training)

        z = conv_output
        for dropout_layer, dense_layer in zip(self.dropout_layers, self.dense_layers):
            d = dropout_layer(z, training=training)
            z = dense_layer(d)
        out_d = z

        # doesn't matter which state_key we use, they're all null padded the same way
        state_key_for_mask = add_planned(self.states_keys[0])
        state_for_mask = input_dict[state_key_for_mask]
        mask = self.mask(state_for_mask)._keras_mask
        out_h = self.lstm(out_d, mask=mask)

        # for every timestep's output, map down to a single scalar, the logit for accept probability
        all_accept_logits = self.output_layer(out_h)
        # ignore the first output, it is meaningless to predict the validity of a single state
        valid_accept_logits = all_accept_logits[:, 1:]
        valid_accept_probabilities = self.sigmoid(valid_accept_logits)

        return {
            'logits': valid_accept_logits,
            'probabilities': valid_accept_probabilities,
            'mask': mask,
        }


class RNNImageClassifierWrapper(BaseConstraintChecker):

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(scenario)
        model_hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.dataset_labeling_params = self.model_hparams['classifier_dataset_hparams']['labeling_params']
        self.horizon = self.dataset_labeling_params['classifier_horizon']
        self.full_env_params = FullEnvParams.from_json(self.model_hparams['classifier_dataset_hparams']['full_env_params'])
        self.net = RNNImageClassifier(hparams=self.model_hparams, batch_size=batch_size, scenario=scenario)
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        self.ckpt.restore(self.manager.latest_checkpoint)

    def check_trajectory(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions) -> tf.Tensor:
        # remove stdev from state we draw
        states_sequence_to_draw = []
        for state in states_sequence:
            states_sequence_to_draw.append({k: state[k] for k in state if k != 'stdev' and k != 'num_diverged'})

        net_inputs = {
            'trajectory_image': image,
            'action': tf.convert_to_tensor(actions, tf.float32),
        }

        if self.net.hparams['stdev']:
            net_inputs[add_planned('stdev')] = tf.convert_to_tensor(states_dict['stdev'], tf.float32)

        for state_key in self.net.states_keys:
            planned_state_key = add_planned(state_key)
            net_inputs[planned_state_key] = tf.convert_to_tensor(states_dict[state_key], tf.float32)

        net_outputs = self.net(add_batch(net_inputs), training=False)[0, 0]
        accept_probability = net_outputs['probabilities']
        return accept_probability

    def check_constraint_differentiable(self,
                                        environment: Dict,
                                        states_sequence: List[Dict],
                                        actions) -> tf.Tensor:
        image_key = self.model_hparams['image_key']
        environment = dict_of_numpy_arrays_to_dict_of_tensors(environment)
        if image_key == 'transition_image':
            raise NotImplementedError()
        elif image_key == 'trajectory_image':
            return self.check_trajectory(environment, states_sequence, actions)
        else:
            raise ValueError('invalid image_key')

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: np.ndarray) -> float:
        actions = tf.Variable(actions, dtype=tf.float32, name="actions")
        states_sequence = [make_dict_float32(s) for s in states_sequence]
        prediction = self.check_constraint_differentiable(environment=environment,
                                                          states_sequence=states_sequence,
                                                          actions=actions)
        return prediction.numpy()


model = RNNImageClassifierWrapper
