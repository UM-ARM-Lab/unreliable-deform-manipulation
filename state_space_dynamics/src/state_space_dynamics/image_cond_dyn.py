import pathlib
from typing import Dict, List

import numpy as np
import rospy
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore
from tensorflow import keras

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.get_local_environment import get_local_env_and_origin_2d_tf
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.get_local_environment import get_local_env_and_origin_3d_tf as get_local_env
from moonshine.raster_3d import raster_3d
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg, send_occupancy_tf
from mps_shape_completion_msgs.msg import OccupancyStamped
from link_bot_pycommon.pycommon import make_dict_tf_float32
from moonshine.matrix_operations import batch_outer_product
from moonshine.moonshine_utils import add_batch, remove_batch, dict_of_sequences_to_sequence_of_dicts_tf, sequence_of_dicts_to_dict_of_tensors
from shape_completion_training.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class ImageCondDynamics(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams, batch_size)
        self.scenario = scenario
        self.initial_epoch = 0

        self.debug_pub = rospy.Publisher('classifier_debug', OccupancyStamped, queue_size=10, latch=True)

        self.rope_image_k = self.hparams['rope_image_k']
        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']

        self.concat = layers.Concatenate()
        self.concat2 = layers.Concatenate()
        # State keys is all the things we want the model to take in/predict
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']

        self.used_states_description = {}
        self.out_dim = 0
        if available_state_name in self.states_keys:
            self.used_states_description[available_state_name] = n
            self.out_dim += n

        self.state_action_dense_layers = []
        if 'state_action_only_fc_layer_sizes' in self.hparams:
            for fc_layer_size in self.hparams['state_action_only_fc_layer_sizes']:
                self.state_action_dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))

        self.conv_only_dense_layers = []
        if 'conv_only_fc_layer_sizes' in self.hparams:
            if len(self.hparams['conv_only_fc_layer_sizes']) > 0:
                for fc_layer_size in self.hparams['conv_only_fc_layer_sizes'][:-1]:
                    self.conv_only_dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))
                self.conv_only_dense_layers.append(layers.Dense(
                    self.hparams['conv_only_fc_layer_sizes'][-1], activation=None))

        self.final_dense_layers = []
        final_fc_layer_sizes = []
        if 'final_fc_layer_sizes' in self.hparams:
            final_fc_layer_sizes = self.hparams['final_fc_layer_sizes']
        elif 'fc_layer_sizes' in self.hparams:
            final_fc_layer_sizes = self.hparams['fc_layer_sizes']
        for fc_layer_size in final_fc_layer_sizes:
            self.final_dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))
        self.final_dense_layers.append(layers.Dense(self.out_dim, activation=None))

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

        # How to combine the image hidden vector and state/action hidden vector
        combine_type = self.hparams.get('combine', 'concat')
        if combine_type == 'outer_product':
            def outer_product_then_flatten(args):
                state_action, conv_z_t = args
                outer_product = batch_outer_product(state_action, conv_z_t)
                return tf.reshape(outer_product, [self.batch_size, -1])

            self.combine_state_action_and_conv_output = layers.Lambda(outer_product_then_flatten)
        elif combine_type == 'concat':
            self.combine_state_action_and_conv_output = layers.Concatenate()

        self.flatten_conv_output = layers.Flatten()

    def compute_loss(self, dataset_element, outputs):
        return {
            'loss': self.scenario.dynamics_loss_function(dataset_element, outputs)
        }

    def calculate_metrics(self, dataset_element, outputs):
        return self.scenario.dynamics_metrics_function(dataset_element, outputs)

    def get_local_env(self, center_point, full_env_origin, full_env, res):
        local_env, local_env_origin = get_local_env_and_origin_2d_tf(center_point=center_point,
                                                                     full_env=full_env,
                                                                     full_env_origin=full_env_origin,
                                                                     res=res,
                                                                     local_h_rows=self.local_env_h_rows,
                                                                     local_w_cols=self.local_env_w_cols)
        return local_env, local_env_origin

    # @tf.function
    def call(self, example, training, mask=None):
        batch_size = example['batch_size']
        time = example['time']

        s_0 = {k: example[k][:, 0] for k in self.state_keys}

        # DEBUG
        # plot the occupancy grid
        time_steps = np.arange(time)
        anim = RvizAnimationController(time_steps)
        b = 0
        full_env_dict = {
            'env': example['env'][b],
            'origin': example['origin'][b],
            'res': example['res'][b],
        }input_dictexample
        self.scenarexample_environment_rviz(full_env_dict)
        # END DEBUG

        pred_states = [s_0]
        for t in range(time):
            s_t = pred_states[-1]
            s_t_local = self.scenario.put_state_local_frame(s_t)
            action_t = {k: example[k][:, t] for k in self.action_keys}
            local_action_t = self.scenario.put_action_local_frame(s_t, action_t)

            local_env_center_t = self.scenario.local_environment_center_differentiable(s_t)

            local_env_t, local_env_origin_t = get_local_env(center_point=local_env_center_t,
                                                            full_env=example['env'],
                                                            full_env_origin=example['origin'],
                                                            res=example['res'],
                                                            local_h_rows=self.local_env_h_rows,
                                                            local_w_cols=self.local_env_w_cols,
                                                            local_c_channels=self.local_env_c_channels,
                                                            batch_size=batch_size)

            concat_args = [tf.expand_dims(local_env_t, axis=4)]
            for state_component_t in s_t.values():
                state_component_voxel_grid = raster_3d(state=state_component_t,
                                                       res=example['res'],
                                                       origin=local_env_origin_t,
                                                       h=self.local_env_h_rows,
                                                       w=self.local_env_w_cols,
                                                       c=self.local_env_c_channels,
                                                       k=self.rope_image_k,
                                                       batch_size=batch_size)
                concat_args.append(state_component_voxel_grid)
            local_voxel_grid_t = tf.concat(concat_args, axis=4)

            # DEBUG
            local_env_dict = {
                'env': tf.clip_by_value(tf.reduce_sum(local_voxel_grid_t[b], axis=-1), 0, 1),
                'origin': local_env_origin_t[b].numpy(),
                'res': example['res'][b].numpy(),
            }
            msg = environment_to_occupancy_msg(local_env_dict, frame='local_occupancy')
            send_occupancy_tf(self.scenario.broadcaster, local_env_dict, frame='local_occupancy')
            self.debug_pub.publish(msg)

            # this will return when either the animation is "playing" or because the user stepped forward
            anim.step()
            # END DEBUG

            # CNN
            z_t = local_voxel_grid_t
            for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
                z_t = conv_layer(z_t)
                z_t = pool_layer(z_t)
            conv_z_t = self.flatten_conv_output(z_t)
            for dense_layer in self.conv_only_dense_layers:
                conv_z_t = dense_layer(conv_z_t)

            # state & action
            state_action_t = self.concat2([s_t_local, local_action_t])
            for dense_layer in self.state_action_dense_layers:
                state_action_t = dense_layer(state_action_t)

            # combine state/action vector with output of CNN
            full_z_t = self.combine_state_action_and_conv_output([state_action_t, conv_z_t])

            # dense layers for combined vector
            for dense_layer in self.final_dense_layers:
                full_z_t = dense_layer(full_z_t)

            delta_s_t = self.vector_to_state_dict(z_t)
            s_t_plus_1 = self.scenario.integrate_dynamics(s_t, delta_s_t)

            pred_states.append(s_t_plus_1)

        return output_states

    def state_vector_to_state_dict(self, s_t):
        state_dict = {}
        start_idx = 0
        for state_key, n in self.used_states_description.items():
            end_idx = start_idx + n
            state_dict[state_key] = s_t[:, start_idx:end_idx]
            start_idx += n
        return state_dict

    def state_vector_to_state_sequence_dict(self, pred_states):
        state_dict = {}
        start_idx = 0
        for state_key, n in self.used_states_description.items():
            end_idx = start_idx + n
            state_dict[state_key] = pred_states[:, :, start_idx:end_idx]
            start_idx += n
        return state_dict


class ImageCondDynamicsWrapper(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(model_dir, batch_size, scenario)
        self.net = ImageCondDynamics(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
        self.ckpt = tf.train.Checkpoint(model=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_dir, max_to_keep=1)

        status = self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
            if self.manager.latest_checkpoint:
                status.assert_existing_objects_matched()
        else:
            raise RuntimeError("Failed to restore!!!")

        self.state_keys = self.net.state_keys
        self.action_keys = self.net.action_keys

    def propagate_from_example(self, dataset_element, training=False):
        return self.net(dataset_element, training=training)

    def propagate_differentiable(self, environment: Dict, start_states: Dict, actions: List[Dict]) -> List[Dict]:
        net_inputs = {k: tf.expand_dims(start_states[k], axis=0) for k in self.state_keys}
        net_inputs.update(sequence_of_dicts_to_dict_of_tensors(actions))
        net_inputs.update(environment)
        net_inputs = add_batch(net_inputs)
        net_inputs = make_dict_tf_float32(net_inputs)

        predictions = self.net((net_inputs, False))
        predictions = remove_batch(predictions)
        predictions = dict_of_sequences_to_sequence_of_dicts_tf(predictions)

        return predictions


model = ImageCondDynamics
