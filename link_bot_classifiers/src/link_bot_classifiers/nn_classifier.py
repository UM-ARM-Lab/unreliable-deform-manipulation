#!/usr/bin/env python
import json
import pathlib
from typing import Dict, List

import numpy as np
import rospy
import tensorflow as tf
from colorama import Fore
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_data.link_bot_dataset_utils import NULL_PAD_VALUE, add_predicted
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg
from link_bot_pycommon.pycommon import make_dict_float32, make_dict_tf_float32
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine import classifier_losses_and_metrics
from moonshine.classifier_losses_and_metrics import \
    binary_classification_sequence_metrics_function
from moonshine.get_local_environment import \
    get_local_env_and_origin_3d_tf as get_local_env
from moonshine.moonshine_utils import (add_batch, remove_batch,
                                       sequence_of_dicts_to_dict_of_tensors)
from moonshine.raster_3d import raster_3d
from mps_shape_completion_msgs.msg import OccupancyStamped
from shape_completion_training.my_keras_model import MyKerasModel
from tensorflow import keras
from tensorflow_core.python.keras import layers
from visualization_msgs.msg import Marker


class NNClassifier(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: Base3DScenario):
        super().__init__(hparams, batch_size)
        self.scenario = scenario

        self.debug_pub = rospy.Publisher('classifier_debug', OccupancyStamped, queue_size=10, latch=True)

        self.classifier_dataset_hparams = self.hparams['classifier_dataset_hparams']
        self.dynamics_dataset_hparams = self.classifier_dataset_hparams['fwd_model_hparams']['dynamics_dataset_hparams']
        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']
        self.rope_image_k = self.hparams['rope_image_k']

        # TODO: add stdev to states keys?
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']

        self.conv_layers = []
        self.pool_layers = []
        for n_filters, kernel_size in self.hparams['conv_filters']:
            conv = layers.Conv3D(n_filters,
                                 kernel_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']))
            pool = layers.MaxPool3D(2)
            self.conv_layers.append(conv)
            self.pool_layers.append(pool)

        if self.hparams['batch_norm']:
            self.batch_norm = layers.BatchNormalization()

        self.dense_layers = []
        for hidden_size in self.hparams['fc_layer_sizes']:
            dense = layers.Dense(hidden_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']))
            self.dense_layers.append(dense)

        self.mask = layers.Masking(mask_value=NULL_PAD_VALUE)
        self.lstm = layers.LSTM(self.hparams['rnn_size'], unroll=True, return_sequences=True)
        self.output_layer = layers.Dense(1, activation=None)
        self.sigmoid = layers.Activation("sigmoid")

        loss_type = self.hparams['loss_type']
        if loss_type == 'weighted_sequence':
            self.loss_function = classifier_losses_and_metrics.negative_weighted_binary_classification_sequence_loss_function
        else:
            raise NotImplementedError()

    def make_traj_voxel_grids_from_input_dict(self, input_dict: Dict, batch_size, time: int):
        conv_outputs = []

        # # DEBUG
        # # plot the occupancy grid
        # time_steps = np.arange(time)
        # anim = RvizAnimationController(time_steps)
        # b = 0
        # full_env_dict = {
        #     'env': input_dict['env'][b],
        #     'origin': input_dict['origin'][b],
        #     'res': input_dict['res'][b],
        # }
        # self.scenario.plot_environment_rviz(full_env_dict)
        # # END DEBUG

        for t in range(time):
            state_t = self.scenario.index_predicted_state_time(input_dict, t)

            local_env_center_t = self.scenario.local_environment_center_differentiable(state_t)

            local_env_t, local_env_origin_t = get_local_env(center_point=local_env_center_t,
                                                            full_env=input_dict['env'][:],
                                                            full_env_origin=input_dict['origin'][:],
                                                            res=input_dict['res'][:],
                                                            local_h_rows=self.local_env_h_rows,
                                                            local_w_cols=self.local_env_w_cols,
                                                            local_c_channels=self.local_env_c_channels,
                                                            batch_size=batch_size)

            concat_args = [tf.expand_dims(local_env_t, axis=4)]
            for state_component_t in state_t.values():
                state_component_voxel_grid = raster_3d(state=state_component_t,
                                                       res=input_dict['res'],
                                                       origin=local_env_origin_t,
                                                       h=self.local_env_h_rows,
                                                       w=self.local_env_w_cols,
                                                       c=self.local_env_c_channels,
                                                       k=self.rope_image_k,
                                                       batch_size=batch_size)
                concat_args.append(state_component_voxel_grid)
            local_voxel_grid_t = tf.concat(concat_args, axis=4)

            # # DEBUG
            # local_env_dict = {
            #     'env': tf.clip_by_value(tf.reduce_sum(local_voxel_grid_t[b], axis=-1), 0, 1),
            #     'origin': local_env_origin_t[b].numpy(),
            #     'res': input_dict['res'][b].numpy(),
            # }
            # msg = environment_to_occupancy_msg(local_env_dict, frame='local_occupancy')
            # link_bot_sdf_utils.send_occupancy_tf(self.scenario.broadcaster, local_env_dict, frame='local_occupancy')
            # self.debug_pub.publish(msg)

            # # this will return when either the animation is "playing" or because the user stepped forward
            # anim.step()
            # # END DEBUG

            conv_z = local_voxel_grid_t
            for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
                conv_h = conv_layer(conv_z)
                conv_z = pool_layer(conv_h)
            out_conv_z = conv_z
            out_conv_z_dim = out_conv_z.shape[1] * out_conv_z.shape[2] * out_conv_z.shape[3] * out_conv_z.shape[4]
            out_conv_z = tf.reshape(out_conv_z, [batch_size, out_conv_z_dim])

            conv_outputs.append(out_conv_z)

        conv_outputs = tf.stack(conv_outputs, axis=1)

        return conv_outputs

    def compute_loss(self, dataset_element, outputs):
        return {
            'loss': self.loss_function(dataset_element, outputs)
        }

    def calculate_metrics(self, dataset_element, outputs):
        return binary_classification_sequence_metrics_function(dataset_element, outputs)

    def call(self, input_dict: Dict, training, **kwargs):
        batch_size = input_dict['batch_size']
        time = input_dict['time']

        conv_output = self.make_traj_voxel_grids_from_input_dict(input_dict, batch_size, time)

        states = {k: input_dict[add_predicted(k)] for k in self.state_keys}
        states = self.scenario.put_state_local_frame(states)
        actions = {k: input_dict[k] for k in self.action_keys}
        all_but_last_states = {k: v[:, :-1] for k, v in states.items()}
        actions = self.scenario.put_action_local_frame(all_but_last_states, actions)
        padded_actions = {k: tf.pad(v, [[0, 0], [0, 1], [0, 0]]) for k, v in actions.items()}
        concat_args = [conv_output] + list(states.values()) + list(padded_actions.values())

        if self.hparams['stdev']:
            stdevs = input_dict[add_predicted('stdev')]
            concat_args.append(stdevs)

        conv_output = tf.concat(concat_args, axis=2)

        if self.hparams['batch_norm']:
            conv_output = self.batch_norm(conv_output, training=training)

        z = conv_output
        for dense_layer in self.dense_layers:
            z = dense_layer(z)
        out_d = z

        # TODO: remove masking, no longer used I believe
        # doesn't matter which state_key we use, they're all null padded the same way
        state_key_for_mask = add_predicted(self.state_keys[0])
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


class NNClassifierWrapper(BaseConstraintChecker):

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: Base3DScenario):
        super().__init__(scenario)
        model_hparams_file = path.parent / 'params.json'
        if not model_hparams_file.exists():
            model_hparams_file = path / 'hparams.json'
            if not model_hparams_file.exists():
                raise FileNotFoundError("no hparams file found!")
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.dataset_labeling_params = self.model_hparams['classifier_dataset_hparams']['labeling_params']
        self.horizon = self.dataset_labeling_params['classifier_horizon']
        self.model = NNClassifier(hparams=self.model_hparams, batch_size=batch_size, scenario=scenario)
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)

        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        else:
            raise RuntimeError("Failed to restore!!!")

    def check_constraint_batched_tf(self,
                                    environment: Dict,
                                    predictions: Dict,
                                    actions: Dict,
                                    batch_size: int,
                                    state_sequence_length: int):
        # construct network inputs
        net_inputs = {
            'batch_size': batch_size,
            'time': state_sequence_length,
        }
        net_inputs.update(make_dict_tf_float32(environment))

        for action_key in self.model.action_keys:
            net_inputs[action_key] = tf.cast(actions[action_key], tf.float32)

        for state_key in self.model.state_keys:
            planned_state_key = add_predicted(state_key)
            net_inputs[planned_state_key] = tf.cast(predictions[state_key], tf.float32)

        if self.model.hparams['stdev']:
            net_inputs[add_predicted('stdev')] = tf.cast(predictions['stdev'], tf.float32)

        predictions = self.model(net_inputs, training=False)
        accept_probabilities = tf.squeeze(predictions['probabilities'], axis=2)
        return accept_probabilities

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        environment = add_batch(environment)
        states_sequence = add_batch(states_sequence)
        states_sequence_dict = sequence_of_dicts_to_dict_of_tensors(states_sequence)
        actions = add_batch(actions)
        accept_probabilities_batched = self.check_constraint_batched_tf(
            environment, states_sequence_dict, actions, 1, len(states_sequence))
        return remove_batch(accept_probabilities_batched)

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        actions = tf.Variable(actions, dtype=tf.float32, name="actions")
        states_sequence = [make_dict_float32(s) for s in states_sequence]
        accept_probabilities = self.check_constraint_tf(environment=environment,
                                                        states_sequence=states_sequence,
                                                        actions=actions)
        accept_probabilities = accept_probabilities.numpy()
        return accept_probabilities


model = NNClassifierWrapper