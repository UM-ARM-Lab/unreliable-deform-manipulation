#!/usr/bin/env python
import json
import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore
from tensorflow import keras

import rospy
from geometry_msgs.msg import TransformStamped
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_classifiers.visualization import visualize_classifier_example_3d
from link_bot_data.link_bot_dataset_utils import add_predicted, NULL_PAD_VALUE
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg
from link_bot_pycommon.pycommon import make_dict_float32
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine import classifier_losses_and_metrics
from moonshine.classifier_losses_and_metrics import binary_classification_sequence_metrics_function
from moonshine.get_local_environment import get_local_env_and_origin_3d_tf as get_local_env
from moonshine.moonshine_utils import add_batch, dict_of_numpy_arrays_to_dict_of_tensors, flatten_batch_and_time, \
    sequence_of_dicts_to_dict_of_sequences, remove_batch, index_dict_of_batched_vectors_tf
from moonshine.raster_3d import raster_3d
from mps_shape_completion_msgs.msg import OccupancyStamped
from shape_completion_training.my_keras_model import MyKerasModel


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
        self.state_keys = self.hparams['states_keys']
        self.action_keys = self.hparams['action_keys']

        self.conv_layers = []
        self.pool_layers = []
        for n_filters, kernel_size in self.hparams['conv_filters']:
            conv = layers.Conv3D(n_filters,
                                 kernel_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 activity_regularizer=keras.regularizers.l1(self.hparams['activity_reg']))
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
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 activity_regularizer=keras.regularizers.l1(self.hparams['activity_reg']))
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

    def make_traj_voxel_grids_from_input_dict(self, input_dict, batch_size, time):
        # First flatten batch & time
        transposed_states_dict = {k: tf.transpose(input_dict[add_predicted(k)], [1, 0, 2]) for k in self.state_keys}
        states_dict_batch_time = flatten_batch_and_time(transposed_states_dict)

        local_env_center_point_batch_time = self.scenario.local_environment_center_differentiable(states_dict_batch_time)
        voxel_grids_batch_and_time, local_env_origins_batch_and_time = self.make_traj_voxel_grids(
            environment=self.scenario.get_environment_from_example(input_dict),
            states_dict_batch_time=states_dict_batch_time,
            local_env_center_point_batch_time=local_env_center_point_batch_time,
            batch_size=batch_size,
            time=time)
        return voxel_grids_batch_and_time, local_env_origins_batch_and_time

    def compute_loss(self, dataset_element, outputs):
        return {
            'loss': self.loss_function(dataset_element, outputs)
        }

    def calculate_metrics(self, dataset_element, outputs):
        return binary_classification_sequence_metrics_function(dataset_element, outputs)

    # @tf.function
    def make_traj_voxel_grids(self,
                              environment,
                              states_dict_batch_time,
                              local_env_center_point_batch_time,
                              batch_size,
                              time):
        """
        :return: [batch, time, h, w, 1 + n_points]
        """
        batch_and_time = batch_size * time
        env_batch_time = tf.tile(environment['env'], [time, 1, 1, 1])
        env_origin_batch_time = tf.tile(environment['origin'], [time, 1])
        env_res_batch_time = tf.tile(environment['res'], [time])

        local_env_batch_time, local_env_origin_batch_time = get_local_env(center_point=local_env_center_point_batch_time,
                                                                          full_env=env_batch_time,
                                                                          full_env_origin=env_origin_batch_time,
                                                                          res=env_res_batch_time,
                                                                          local_h_rows=self.local_env_h_rows,
                                                                          local_w_cols=self.local_env_w_cols,
                                                                          local_c_channels=self.local_env_c_channels)

        concat_args = []
        for planned_state in states_dict_batch_time.values():
            planned_rope_voxel_grid = raster_3d(state=planned_state,
                                                res=env_res_batch_time,
                                                origin=local_env_origin_batch_time,
                                                h=self.local_env_h_rows,
                                                w=self.local_env_w_cols,
                                                c=self.local_env_c_channels,
                                                k=self.rope_image_k,
                                                batch_size=batch_and_time)
            concat_args.append(planned_rope_voxel_grid)

        concat_args.append(tf.expand_dims(local_env_batch_time, axis=4))
        voxel_grids_batch_time = tf.concat(concat_args, axis=4)
        return voxel_grids_batch_time, local_env_origin_batch_time

    def _conv(self, voxel_grids, time, batch_size):
        # merge batch & time dimensions
        conv_z = voxel_grids
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z
        out_conv_z = tf.reshape(out_conv_z, [time, batch_size, -1])
        # un-merge batch & time dimensions

        return out_conv_z

    def debug_plot(self, input_dict, voxel_grids, local_env_origins, time):
        # plot the occupancy grid
        debug_env = tf.clip_by_value(tf.reduce_sum(voxel_grids, axis=5), 0, 1)
        time_steps = np.arange(voxel_grids.shape[1])
        anim = RvizAnimationController(time_steps)
        b = 0
        while not anim.done:
            t = anim.t()
            environment = {
                'env': debug_env[b, t],
                'res': input_dict['res'][b],
                'origin': local_env_origins[b, t],
            }

            static_transformStamped = TransformStamped()
            static_transformStamped.header.stamp = rospy.Time.now()
            static_transformStamped.header.frame_id = "world"
            static_transformStamped.child_frame_id = "local_occupancy"
            origin_x, origin_y, origin_z = link_bot_sdf_utils.idx_to_point_3d_in_env(0, 0, 0, environment)
            static_transformStamped.transform.translation.x = origin_x
            static_transformStamped.transform.translation.y = origin_y
            static_transformStamped.transform.translation.z = origin_z
            static_transformStamped.transform.rotation.x = 0
            static_transformStamped.transform.rotation.y = 0
            static_transformStamped.transform.rotation.z = 0
            static_transformStamped.transform.rotation.w = 1
            self.scenario.broadcaster.sendTransform(static_transformStamped)

            msg = environment_to_occupancy_msg(environment, frame='local_occupancy')
            self.debug_pub.publish(msg)

            # this will return when either the animation is "playing" or because the user stepped forward
            anim.step()

        input_dict.pop('batch_size')
        visualize_classifier_example_3d(scenario=self.scenario,
                                        example=index_dict_of_batched_vectors_tf(input_dict, 0),
                                        n_time_steps=time)

    # @tf.function
    def call(self, input_dict: Dict, training, **kwargs):
        # FIXME: how to get time/batch size without just knowing a using key that happens to have the right shape?
        batch_size = input_dict['batch_size']
        time = input_dict['time'][0]
        voxel_grids_batch_and_time, local_env_origins_batch_and_time = self.make_traj_voxel_grids_from_input_dict(input_dict, batch_size, time)

        voxel_grids = tf.reshape(voxel_grids_batch_and_time, [time, batch_size, self.local_env_h_rows, self.local_env_w_cols, self.local_env_c_channels, -1])
        local_env_origins = tf.reshape(local_env_origins_batch_and_time, [time, batch_size, 3])
        local_env_origins = tf.transpose(local_env_origins, [1, 0, 2])  # undo transpose
        voxel_grids = tf.transpose(voxel_grids, [1, 0, 2, 3, 4, 5])  # undo transpose
        conv_output = self._conv(voxel_grids_batch_and_time, time, batch_size)
        conv_output = tf.transpose(conv_output, [1, 0, 2])  # undo transpose

        states = {k: input_dict[add_predicted(k)] for k in self.state_keys}
        states = self.scenario.put_state_local_frame(states)
        actions = {k: input_dict[k] for k in self.action_keys}
        all_but_last_states = {k: v[:, :-1] for k, v in states.items()}
        actions = self.scenario.put_action_local_frame(all_but_last_states, actions)
        concat_args = [conv_output] + list(states.values()) + list(actions.values())

        if self.hparams['stdev']:
            stdevs = input_dict[add_predicted('stdev')]
            concat_args.append(stdevs)

        # uncomment to debug, also comment out the tf.function's
        self.debug_plot(input_dict, voxel_grids, local_env_origins, time)

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
            'voxel_grids': voxel_grids,
        }


class NNClassifierWrapper(BaseConstraintChecker):

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(scenario)
        model_hparams_file = path / 'hparams.json'
        if not model_hparams_file.exists():
            model_hparams_file = path.parent / 'params.json'
            if not model_hparams_file.exists():
                raise FileNotFoundError("no hparams file found!")
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.dataset_labeling_params = self.model_hparams['classifier_dataset_hparams']['labeling_params']
        self.horizon = self.dataset_labeling_params['classifier_horizon']
        self.net = NNClassifier(hparams=self.model_hparams, batch_size=batch_size, scenario=scenario)
        self.ckpt = tf.train.Checkpoint(model=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)

        status = self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
            if self.manager.latest_checkpoint:
                status.assert_existing_objects_matched()
        else:
            raise RuntimeError("Failed to restore!!!")

    def check_constraint_differentiable_batched_tf(self,
                                                   environment: Dict,
                                                   predictions: Dict,
                                                   actions) -> tf.Tensor:
        # construct network inputs
        net_inputs = {
            'action': tf.convert_to_tensor(actions, tf.float32),
        }
        net_inputs.update(environment)

        if self.net.hparams['stdev']:
            net_inputs[add_predicted('stdev')] = tf.convert_to_tensor(predictions['stdev'], tf.float32)

        for state_key in self.net.state_keys:
            planned_state_key = add_predicted(state_key)
            net_inputs[planned_state_key] = tf.convert_to_tensor(predictions[state_key], tf.float32)

        predictions = self.net(net_inputs, training=False)
        accept_probabilities = tf.squeeze(predictions['probabilities'], axis=2)
        return accept_probabilities

    def check_constraint_differentiable(self,
                                        environment: Dict,
                                        states_sequence: List[Dict],
                                        actions) -> tf.Tensor:
        environment = dict_of_numpy_arrays_to_dict_of_tensors(environment)
        # construct network inputs
        states_sequences_dict = sequence_of_dicts_to_dict_of_sequences(states_sequence)
        net_inputs = {
            'action': tf.convert_to_tensor(actions, tf.float32),
        }
        net_inputs.update(environment)

        if self.net.hparams['stdev']:
            net_inputs[add_predicted('stdev')] = tf.convert_to_tensor(states_sequences_dict['stdev'], tf.float32)

        for state_key in self.net.state_keys:
            planned_state_key = add_predicted(state_key)
            net_inputs[planned_state_key] = tf.convert_to_tensor(states_sequences_dict[state_key], tf.float32)

        predictions = remove_batch(self.net(add_batch(net_inputs), training=False))
        accept_probabilities = tf.squeeze(predictions['probabilities'], axis=1)
        return accept_probabilities

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: np.ndarray):
        actions = tf.Variable(actions, dtype=tf.float32, name="actions")
        states_sequence = [make_dict_float32(s) for s in states_sequence]
        accept_probabilities = self.check_constraint_differentiable(environment=environment,
                                                                    states_sequence=states_sequence,
                                                                    actions=actions)
        accept_probabilities = accept_probabilities.numpy()
        return accept_probabilities


model = NNClassifierWrapper
