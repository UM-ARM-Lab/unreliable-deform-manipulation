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
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_data.link_bot_dataset_utils import add_predicted, NULL_PAD_VALUE
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg
from link_bot_pycommon.pycommon import make_dict_float32, make_dict_tf_float32
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine import classifier_losses_and_metrics
from moonshine.classifier_losses_and_metrics import binary_classification_sequence_metrics_function
from moonshine.get_local_environment import get_local_env_and_origin_3d_tf as get_local_env
from moonshine.moonshine_utils import add_batch, flatten_batch_and_time, \
    remove_batch
from moonshine.raster_3d import raster_3d
from mps_shape_completion_msgs.msg import OccupancyStamped
from shape_completion_training.my_keras_model import MyKerasModel
from visualization_msgs.msg import Marker


class NNClassifier(MyKerasModel):
    def __init__(self, hparams: Dict, batch_size: int, scenario: Base3DScenario):
        super().__init__(hparams, batch_size)
        self.scenario = scenario

        self.debug_pub = rospy.Publisher('classifier_debug', OccupancyStamped, queue_size=10, latch=True)
        self.debug_pub2 = rospy.Publisher('classifier_debug2', Marker, queue_size=10, latch=True)

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

        # self.debug2(local_env_center_point_batch_time, time)

        voxel_grids_batch_and_time, local_env_origins_batch_and_time = self.make_traj_voxel_grids(
            environment=self.scenario.get_environment_from_example(input_dict),
            predicted_state_batch_time=states_dict_batch_time,
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

    def make_traj_voxel_grids(self,
                              environment,
                              predicted_state_batch_time,
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
                                                                          local_c_channels=self.local_env_c_channels,
                                                                          batch_size=batch_and_time)

        concat_args = []
        for predicted_state in predicted_state_batch_time.values():
            planned_rope_voxel_grid = raster_3d(state=predicted_state,
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
        # un-merge batch & time dimensions
        out_conv_z_dim = out_conv_z.shape[1] * out_conv_z.shape[2] * out_conv_z.shape[3] * out_conv_z.shape[4]
        out_conv_z = tf.reshape(out_conv_z, [time, batch_size, out_conv_z_dim])

        return out_conv_z

    def debug_plot(self, input_dict, voxel_grids, local_env_origins, time):
        # plot the occupancy grid
        input_dict.pop('batch_size')
        time_steps = np.arange(time)
        anim = RvizAnimationController(time_steps)
        b = 0
        full_env = {
            'env': input_dict['env'][b],
            'res': input_dict['res'][b],
            'origin': input_dict['origin'][b],
        }
        self.scenario.plot_environment_rviz(full_env)
        while not anim.done:
            t = anim.t()
            environment = {
                'env': tf.cast(tf.reduce_any(voxel_grids[b, t] > 0.5, axis=3), tf.float32),
                'res': input_dict['res'][b],
                'origin': local_env_origins[b, t],
            }

            # msg = voxel_grid_to_colored_point_cloud(voxel_grids[b, t], environment, frame='world')
            msg = environment_to_occupancy_msg(environment, frame='local_occupancy')
            link_bot_sdf_utils.send_occupancy_tf(self.scenario.broadcaster, environment, frame='local_occupancy')
            self.debug_pub.publish(msg)

            actual_t = remove_batch(self.scenario.index_state_time(input_dict, t))
            pred_t = remove_batch(self.scenario.index_predicted_state_time(input_dict, t))
            action_t = remove_batch(self.scenario.index_action_time(input_dict, t))
            label_t = remove_batch(self.scenario.index_label_time(input_dict, t)).numpy()
            self.scenario.plot_state_rviz(actual_t, label='actual', color='#ff0000aa')
            self.scenario.plot_state_rviz(pred_t, label='predicted', color='#0000ffaa')
            self.scenario.plot_action_rviz(actual_t, action_t)
            self.scenario.plot_is_close(label_t)

            # this will return when either the animation is "playing" or because the user stepped forward
            anim.step()

    def debug2(self, local_env_center_point_batch_time, time):
        for i in range(time):
            # for i in range(1):
            m = Marker()
            m.action = Marker.ADD
            m.header.frame_id = 'world'
            m.type = Marker.CUBE
            m.scale.x = 0.01
            m.scale.y = 0.01
            m.scale.z = 0.01
            m.id = i
            m.pose.position.x = local_env_center_point_batch_time[i, 0]
            m.pose.position.y = local_env_center_point_batch_time[i, 1]
            m.pose.position.z = local_env_center_point_batch_time[i, 2]
            m.pose.orientation.w = 1
            m.color.r = 255
            m.color.g = 255
            m.color.a = 255
            self.debug_pub2.publish(m)

    @tf.function
    def call(self, input_dict: Dict, training, **kwargs):
        batch_size = input_dict['batch_size']
        time = input_dict['time'][0]

        voxel_grids_batch_and_time, local_env_origins_batch_and_time = self.make_traj_voxel_grids_from_input_dict(input_dict,
                                                                                                                  batch_size,
                                                                                                                  time)

        voxel_grids = tf.reshape(voxel_grids_batch_and_time,
                                 [time, batch_size, self.local_env_h_rows, self.local_env_w_cols, self.local_env_c_channels, -1])
        local_env_origins = tf.reshape(local_env_origins_batch_and_time, [time, batch_size, 3])
        local_env_origins = tf.transpose(local_env_origins, [1, 0, 2])  # undo transpose
        voxel_grids = tf.transpose(voxel_grids, [1, 0, 2, 3, 4, 5])  # undo transpose
        conv_output = self._conv(voxel_grids_batch_and_time, time, batch_size)
        conv_output = tf.transpose(conv_output, [1, 0, 2])  # undo transpose

        # uncomment to debug, also comment out the tf.function's
        # self.debug_plot(input_dict, voxel_grids, local_env_origins, time)

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
            'voxel_grids': voxel_grids,
        }


class NNClassifierWrapper(BaseConstraintChecker):

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: Base3DScenario):
        super().__init__(scenario)
        model_hparams_file = path / 'hparams.json'
        if not model_hparams_file.exists():
            model_hparams_file = path.parent / 'params.json'
            if not model_hparams_file.exists():
                raise FileNotFoundError("no hparams file found!")
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.dataset_labeling_params = self.model_hparams['classifier_dataset_hparams']['labeling_params']
        self.horizon = self.dataset_labeling_params['classifier_horizon']
        self.model = NNClassifier(hparams=self.model_hparams, batch_size=batch_size, scenario=scenario)
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)

        status = self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        else:
            raise RuntimeError("Failed to restore!!!")

    def check_constraint_batched_tf(self,
                                    environment: Dict,
                                    predictions: Dict,
                                    actions: Dict,
                                    state_sequence_length: int):
        # construct network inputs
        net_inputs = {
            'batch_size': tf.constant(1, dtype=tf.int64),
            'time': tf.cast([state_sequence_length], tf.int64),
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
        actions = add_batch(actions)
        accept_probabilities_batched = self.check_constraint_batched_tf(environment, states_sequence, actions)
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
