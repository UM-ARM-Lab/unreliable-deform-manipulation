import pathlib
from typing import Dict, List

import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import add_batch, remove_batch, dict_of_sequences_to_sequence_of_dicts_tf, \
    sequence_of_dicts_to_dict_of_tensors
from shape_completion_training.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class UnconstrainedDynamicsNN(MyKerasModel):

    def compute_loss(self, example, outputs):
        return {
            'loss': self.scenario.dynamics_loss_function(example, outputs)
        }

    def compute_metrics(self, example, outputs):
        return self.scenario.dynamics_metrics_function(example, outputs)

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.scenario = scenario
        self.initial_epoch = 0

        self.concat = layers.Concatenate()
        self.dense_layers = []
        for fc_layer_size in self.hparams['fc_layer_sizes']:
            self.dense_layers.append(layers.Dense(fc_layer_size, activation='relu', use_bias=True))

        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']
        self.dataset_states_description = self.hparams['dynamics_dataset_hparams']['states_description']
        self.dataset_actions_description = self.hparams['dynamics_dataset_hparams']['action_description']
        self.state_dimensions = [self.dataset_states_description[k] for k in self.state_keys]
        self.total_state_dimensions = sum(self.state_dimensions)

        self.dense_layers.append(layers.Dense(self.total_state_dimensions, activation=None))

    def debug_plot(self, s):
        self.scenario.plot_state_rviz({'link_bot': s['link_bot'][0]})

    @tf.function
    def call(self, example, training, mask=None):
        actions = {k: example[k] for k in self.action_keys}
        input_sequence_length = actions[self.action_keys[0]].shape[1]
        s_0 = {k: example[k][:, 0] for k in self.state_keys}

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]
            action_t = {k: a[:, t] for k, a in actions.items()}

            if self.hparams['use_local_frame']:
                s_t_local = self.scenario.put_state_local_frame(s_t)
                # self.debug_plot(s_t_local)
                states_and_actions = list(s_t_local.values()) + list(action_t.values())
            else:
                states_and_actions = list(s_t.values()) + list(action_t.values())

            # concat into one big state-action vector
            z_t = self.concat(states_and_actions)
            for dense_layer in self.dense_layers:
                z_t = dense_layer(z_t)

            if self.hparams['residual']:
                delta_s_t = self.vector_to_state_dict(z_t)
                s_t_plus_1 = self.scenario.integrate_dynamics(s_t, delta_s_t)
            else:
                s_t_plus_1 = self.vector_to_state_dict(z_t)

            pred_states.append(s_t_plus_1)

        pred_states_dict = sequence_of_dicts_to_dict_of_tensors(pred_states, axis=1)
        return pred_states_dict

    def vector_to_state_dict(self, z):
        start_idx = 0
        state_vectors = []
        for dim in self.state_dimensions:
            state_vectors.append(z[:, start_idx:start_idx + dim])
            start_idx += dim
        return dict(zip(self.state_keys, state_vectors))


class UDNNWrapper(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(model_dir, batch_size, scenario)
        self.net = UnconstrainedDynamicsNN(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
        self.ckpt = tf.train.Checkpoint(model=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_dir, max_to_keep=1)

        status = self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
            if self.manager.latest_checkpoint:
                status.assert_existing_objects_matched()
        else:
            raise RuntimeError("Failed to restore!!!")
        self.states_keys = [self.net.state_key]

    def propagate_from_example(self, example, training=False):
        return self.net(example, training=training)

    def propagate_differentiable(self, environment: Dict, start_states: Dict, actions) -> List[Dict]:
        del environment  # unused
        state = start_states[self.net.state_key]
        state = tf.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        test_x = {
            # must be batch, T, n_state
            self.net.state_key: state,
            # must be batch, T, n_action
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
