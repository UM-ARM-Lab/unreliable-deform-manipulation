import pathlib
from typing import Dict

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore

from moonshine.numpy_utils import add_batch_to_dict
from moonshine.tensorflow_train_test_loop import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class SimpleNN(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.initial_epoch = 0

        self.concat = layers.Concatenate()
        self.dense_layers = []
        for fc_layer_size in self.hparams['fc_layer_sizes']:
            self.dense_layers.append(layers.Dense(fc_layer_size, activation='relu', use_bias=True))
        # TODO: make state_key always mean without "state/" and state_feature always mean with
        self.state_key = self.hparams['state_key']
        self.state_feature = "state/{}".format(self.state_key)
        self.n_state = self.hparams['dynamics_dataset_hparams']['states_description'][self.state_key]
        self.dense_layers.append(layers.Dense(self.n_state, activation=None))

    def call(self, dataset_element, training=None, mask=None):
        input_dict, _ = dataset_element
        states = input_dict[self.state_feature]
        actions = input_dict['action']
        input_sequence_length = actions.shape[1]
        s_0 = tf.expand_dims(states[:, 0], axis=2)

        gen_states = [s_0]
        for t in range(input_sequence_length):
            s_t = gen_states[-1]
            action_t = actions[:, t]

            s_t_squeeze = tf.squeeze(s_t, squeeze_dims=2)
            _state_action_t = self.concat([s_t_squeeze, action_t])
            z_t = _state_action_t
            for dense_layer in self.dense_layers:
                z_t = dense_layer(z_t)

            if self.hparams['residual']:
                ds_t = tf.expand_dims(z_t, axis=2)
                s_t_plus_1_flat = s_t + ds_t
            else:
                s_t_plus_1_flat = tf.expand_dims(z_t, axis=2)

            gen_states.append(s_t_plus_1_flat)

        gen_states = tf.stack(gen_states)
        gen_states = tf.transpose(gen_states, [1, 0, 2, 3])
        gen_states = tf.squeeze(gen_states, squeeze_dims=3)
        return {self.state_feature: gen_states}


class SimpleNNWrapper(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path, batch_size: int):
        super().__init__(model_dir, batch_size)
        self.net = SimpleNN(hparams=self.hparams, batch_size=batch_size)
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, model_dir, max_to_keep=1)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        self.state_keys = [self.net.state_key]

    def propagate(self,
                  full_env: np.ndarray,
                  full_env_origin: np.ndarray,
                  res: np.ndarray,
                  states: np.ndarray,
                  actions: np.ndarray) -> np.ndarray:
        del full_env  # unused
        del full_env_origin  # unused
        del res  # unsed
        state = states[self.net.state_key]
        T, _ = actions.shape
        state = np.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        test_x = {
            # must be batch, T, n_state
            self.net.state_feature: state,
            # must be batch, T, 2
            'action': actions,
        }
        test_x = add_batch_to_dict(test_x)
        predictions = self.net(test_x)

        for k, v in predictions.items():
            predictions[k] = np.reshape(v.numpy(), [T + 1, -1]).astype(np.float64)

        return predictions
