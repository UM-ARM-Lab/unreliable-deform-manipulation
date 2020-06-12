import pathlib
from typing import Dict, List

import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import add_batch, remove_batch, dict_of_sequences_to_sequence_of_dicts_tf
from shape_completion_training.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class UnconstrainedDynamicsNN(MyKerasModel):

    def compute_loss(self, dataset_element, outputs):
        return {
            'loss': self.scenario.dynamics_loss_function(dataset_element, outputs)
        }

    def compute_metrics(self, dataset_element, outputs):
        return self.scenario.dynamics_metrics_function(dataset_element, outputs)

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.scenario = scenario
        self.initial_epoch = 0

        self.concat = layers.Concatenate()
        self.dense_layers = []
        for fc_layer_size in self.hparams['fc_layer_sizes']:
            self.dense_layers.append(layers.Dense(fc_layer_size, activation='relu', use_bias=True))
        self.state_key = self.hparams['state_key']
        # TODO: support multiple state keys
        self.n_state = self.hparams['dynamics_dataset_hparams']['states_description'][self.state_key]
        self.dense_layers.append(layers.Dense(self.n_state, activation=None))

    def debug_plot(self, s):
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.gca()
        self.scenario.plot_state(ax, {'link_bot': s[0]})
        plt.axis("equal")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show(block=True)

    @tf.function()
    def call(self, dataset_element, training, mask=None):
        input_dict, _ = dataset_element
        states = input_dict[self.state_key]
        actions = input_dict['action']
        input_sequence_length = actions.shape[1]
        s_0 = states[:, 0]

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]
            action_t = actions[:, t]

            if self.hparams['use_local_frame']:
                s_t_w_time = tf.expand_dims(s_t, axis=1)
                s_t_local = self.scenario.put_state_local_frame(self.state_key, s_t_w_time)
                s_t_local = tf.squeeze(s_t_local, axis=1)
                _state_action_t = self.concat([s_t_local, action_t])
            else:
                _state_action_t = self.concat([s_t, action_t])
            z_t = _state_action_t
            for dense_layer in self.dense_layers:
                z_t = dense_layer(z_t)

            if self.hparams['residual']:
                ds_t = z_t
                s_t_plus_1_flat = self.scenario.integrate_dynamics(s_t, ds_t)
            else:
                s_t_plus_1_flat = z_t

            pred_states.append(s_t_plus_1_flat)

        pred_states = tf.stack(pred_states, axis=1)
        return {self.state_key: pred_states}


class SimpleNNWrapper(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(model_dir, batch_size, scenario)
        self.net = UnconstrainedDynamicsNN(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
        self.ckpt = tf.train.Checkpoint(net=self.net, model=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_dir, max_to_keep=1)
        status = self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
            if self.manager.latest_checkpoint:
                status.assert_existing_objects_matched()
        else:
            raise RuntimeError("Failed to restore!!!")
        self.states_keys = [self.net.state_key]

    def propagate_from_dataset_element(self, dataset_element, training=False):
        return self.net(dataset_element, training=training)

    def propagate_differentiable(self,
                                 environment: Dict,
                                 start_states: Dict,
                                 actions) -> List[Dict]:
        """
        :param start_states:          each value in the dictionary should be of shape (batch, n_state)
        :param actions:        (T, 2)
        :return: states:       each value in the dictionary should be a of shape [batch, T+1, n_state)
        """
        del environment  # unused
        state = start_states[self.net.state_key]
        state = tf.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        test_x = {
            # must be batch, T, n_state
            self.net.state_key: state,
            # must be batch, T, 2
            'action': actions,
        }
        test_x = add_batch(test_x)
        # the network returns a dictionary where each value is [T, n_state]
        # which is what you'd want for training, but for planning and execution and everything else
        # it is easier to deal with a list of states where each state is a dictionary
        predictions = self.net((test_x, None), training=False)
        predictions = remove_batch(predictions)
        predictions = dict_of_sequences_to_sequence_of_dicts_tf(predictions)
        return predictions
