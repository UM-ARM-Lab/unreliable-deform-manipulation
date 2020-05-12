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
from link_bot_data.link_bot_dataset_utils import add_next_and_planned, add_planned
from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_planning.params import FullEnvParams
from link_bot_pycommon.link_bot_pycommon import make_dict_float32
from moonshine.image_functions import make_transition_images, make_traj_images_from_states_list
from moonshine.moonshine_utils import add_batch, dict_of_numpy_arrays_to_dict_of_tensors
from moonshine.tensorflow_train_test_loop import MyKerasModel


class RNNImageClassifier(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams, batch_size, scenario=scenario)

        self.classifier_dataset_hparams = self.hparams['classifier_dataset_hparams']
        self.dynamics_dataset_hparams = self.classifier_dataset_hparams['fwd_model_hparams']['dynamics_dataset_hparams']
        self.n_action = self.dynamics_dataset_hparams['n_action']
        self.batch_size = batch_size

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

        self.lstm = layers.LSTM(self.hparams['rnn_size'], unroll=True)
        self.output_layer = layers.Dense(1, activation='sigmoid')

    # @tf.function
    def _conv(self, images):
        # merge batch & time dimensions
        batch, time, h, w, c = images.shape
        conv_z = tf.reshape(images, [batch * time, h, w, c])
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z
        out_conv_z = tf.reshape(out_conv_z, [batch, time, -1])
        print(out_conv_z.shape)
        # un-merge batch & time dimensions

        return out_conv_z

    # @tf.function
    def call(self, input_dict: Dict, training=True, mask=None):
        # Choose what key to use, so depending on how the model was trained it will expect a transition_image or trajectory_image
        images = input_dict[self.hparams['image_key']]

        action = input_dict['action']
        conv_output = self._conv(images)

        concat_args = [conv_output, action]
        if self.hparams['stdev']:
            stdevs = input_dict[add_planned('stdev')]
            concat_args.append(stdevs)
        for state_key in self.states_keys:
            planned_state_key = add_planned(state_key)
            state = input_dict[planned_state_key]
            concat_args.append(state)
        print(concat_args)
        conv_output = tf.concat(concat_args, axis=2)
        print(conv_output.shape)

        if self.hparams['batch_norm']:
            conv_output = self.batch_norm(conv_output, training=training)

        z = conv_output
        for dropout_layer, dense_layer in zip(self.dropout_layers, self.dense_layers):
            d = dropout_layer(z, training=training)
            z = dense_layer(d)
        out_d = z
        print(out_d.shape)

        out_h = self.lstm(out_d)
        print(out_h.shape)
        accept_probability = self.output_layer(out_h)
        print(accept_probability.shape)
        return accept_probability


class RNNImageClassifierWrapper(BaseConstraintChecker):

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(scenario)
        model_hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.full_env_params = FullEnvParams.from_json(self.model_hparams['classifier_dataset_hparams']['full_env_params'])
        self.input_h_rows = self.model_hparams['input_h_rows']
        self.input_w_cols = self.model_hparams['input_w_cols']
        self.net = RNNImageClassifier(hparams=self.model_hparams, batch_size=batch_size, scenario=scenario)
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        self.ckpt.restore(self.manager.latest_checkpoint)

    def check_transition(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions,
                         ) -> tf.Tensor:
        states_i = states_sequence[-2]
        # remove stdev from state we draw. if stdev doesn't exist this will still work
        states_i_to_draw = {k: states_i[k] for k in states_i if k != 'stdev'}
        action_i = actions[-1]
        states_i_plus_1 = states_sequence[-1]
        states_i_plus_1_to_draw = {k: states_i_plus_1[k] for k in states_i_plus_1 if k != 'stdev'}

        action_in_image = self.model_hparams['action_in_image']
        batched_inputs = add_batch(environment, states_i_to_draw, action_i, states_i_plus_1_to_draw)
        image = make_transition_images(*batched_inputs,
                                       scenario=self.scenario,
                                       local_env_h=self.input_h_rows,
                                       local_env_w=self.input_w_cols,
                                       action_in_image=action_in_image,
                                       batch_size=1,
                                       k=self.model_hparams['rope_image_k'])[0]
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        net_inputs = self.net_inputs(action_i, states_i, states_i_plus_1)
        net_inputs['transition_image'] = image

        accept_probability = self.net(add_batch(net_inputs), training=False)[0, 0]
        return accept_probability

    def check_trajectory(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: tf.Variable) -> tf.Tensor:
        # Get state states/action for just the transition, which we also feed into the classifier
        action_i = actions[-1]
        states_i = states_sequence[-2]
        states_i_plus_1 = states_sequence[1]

        # remove stdev from state we draw
        states_sequence_to_draw = []
        for state in states_sequence:
            states_sequence_to_draw.append({k: state[k] for k in state if k != 'stdev'})

        batched_inputs = add_batch(environment, states_sequence_to_draw)
        image = make_traj_images_from_states_list(*batched_inputs, rope_image_k=self.model_hparams['rope_image_k'])[0]

        net_inputs = self.net_inputs(action_i, states_i, states_i_plus_1)
        net_inputs['trajectory_image'] = image

        accept_probability = self.net(add_batch(net_inputs), training=False)[0, 0]
        return accept_probability

    def check_constraint_differentiable(self,
                                        environment: Dict,
                                        states_sequence: List[Dict],
                                        actions) -> tf.Tensor:
        image_key = self.model_hparams['image_key']
        environment = dict_of_numpy_arrays_to_dict_of_tensors(environment)
        if image_key == 'transition_image':
            return self.check_transition(environment=environment,
                                         states_sequence=states_sequence,
                                         actions=actions)
        elif image_key == 'trajectory_image':
            return self.check_trajectory(environment, states_sequence, actions)
        else:
            raise ValueError('invalid image_key')

    def check_constraint(self,
                         environement: Dict,
                         states_sequence: List[Dict],
                         actions: np.ndarray) -> float:
        actions = tf.Variable(actions, dtype=tf.float32, name="actions")
        states_sequence = [make_dict_float32(s) for s in states_sequence]
        prediction = self.check_constraint_differentiable(environment=environement,
                                                          states_sequence=states_sequence,
                                                          actions=actions)
        return prediction.numpy()

    def net_inputs(self, action_i, states_i, states_i_plus_1):
        net_inputs = {
            'action': tf.convert_to_tensor(action_i, tf.float32),
        }

        if self.net.hparams['stdev']:
            net_inputs[add_planned('stdev')] = tf.convert_to_tensor(states_i['stdev'], tf.float32)
            net_inputs[add_next_and_planned('stdev')] = tf.convert_to_tensor(states_i_plus_1['stdev'], tf.float32)

        for state_key in self.net.states_keys:
            planned_state_key = add_planned(state_key)
            planned_state_key_next = add_next_and_planned(state_key)
            net_inputs[planned_state_key] = tf.convert_to_tensor(states_i[state_key], tf.float32)
            net_inputs[planned_state_key_next] = tf.convert_to_tensor(states_i_plus_1[state_key], tf.float32)

        return net_inputs


model = RNNImageClassifierWrapper