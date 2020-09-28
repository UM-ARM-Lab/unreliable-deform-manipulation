from typing import Dict, List

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.models import Sequential

from link_bot_data.link_bot_dataset_utils import add_positive
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from shape_completion_training.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class CFMNetwork(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.scenario = scenario

        self.encoder = Encoder(z_dim=self.hparams['z_dim'], channel_dim=4)
        self.predictor = LocallyLinearPredictor(z_dim=self.hparams['z_dim'],
                                                action_dim=self.hparams['action_dim'],
                                                hidden_sizes=self.hparams['fc_layer_sizes'])

        # self.concat = layers.Concatenate()
        # self.dense_layers = []
        # for fc_layer_size in self.hparams['fc_layer_sizes']:
        #     self.dense_layers.append(layers.Dense(fc_layer_size, activation='relu', use_bias=True))
        #
        # self.state_keys = self.hparams['state_keys']
        # self.action_keys = self.hparams['action_keys']
        # self.dataset_states_description = self.hparams['dynamics_dataset_hparams']['states_description']
        # self.dataset_actions_description = self.hparams['dynamics_dataset_hparams']['action_description']
        # self.state_dimensions = [self.dataset_states_description[k] for k in self.state_keys]
        # self.total_state_dimensions = sum(self.state_dimensions)
        #
        # self.dense_layers.append(layers.Dense(self.total_state_dimensions, activation=None))

    @tf.function
    def call(self, example, training, mask=None):
        actions = {k: example[k] for k in self.action_keys}
        obs = {k: example[k] for k in self.state_keys}
        obs_pos = {k: example[add_positive(k)] for k in self.state_keys}

        # forward pass
        z, z_pos = self.encoder(obs), self.encoder(obs_pos)  # b x z_dim
        z_next = self.predictor(z, actions)

        return {
            'z': z,
            'z_pos': z_pos,
            'z_next': z_next,
        }

    def compute_loss(self, example, outputs):
        batch_size = example['batch_size']
        z = example['z']
        z_pos = example['z_pos']
        z_next = outputs['z_next']

        # loss
        neg_dot_products = tf.linalg.matmul(z_next, z.t())  # b x b
        neg_dists = -((z_next ** 2).sum(1).unsqueeze(1) - 2 * neg_dot_products + (z ** 2).sum(1).unsqueeze(0))
        idxs = tf.range(batch_size)

        # Set to minus infinity entries when comparing z with z - will be zero when apply softmax
        neg_dists[idxs, idxs] = float('-inf')  # b x b+1

        pos_dot_products = (z_pos * z_next).sum(axis=1)  # b
        pos_dists = -((z_pos ** 2).sum(1) - 2 * pos_dot_products + (z_next ** 2).sum(1))
        pos_dists = pos_dists.unsqueeze(1)  # b x 1

        dists = tf.concat((neg_dists, pos_dists), axis=1)  # b x b+1
        dists = tf.log_softmax(dists, dim=1)  # b x b+1
        loss = -dists[:, -1].mean()  # Get last column with is the true pos sample

        return {
            'loss': loss
        }

    def calculate_metrics(self, example, outputs):
        metrics = self.scenario.dynamics_metrics_function(example, outputs)
        metrics['loss'] = self.scenario.dynamics_loss_function(example, outputs)
        return metrics


class CFMWrapper(BaseDynamicsFunction):

    @staticmethod
    def get_net_class():
        return CFMNetwork


class Encoder(tf.keras.Model):

    def __init__(self, z_dim, channel_dim):
        super().__init__()

        self.z_dim = z_dim
        self.model = Sequential(
            layers.Conv2D(channel_dim, 64, 3, 1, 1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(64, 64, 4, 2, 1),
            layers.LeakyReLU(0.2),
            # 64 x 32 x 32
            layers.Conv2D(64, 64, 3, 1, 1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(64, 128, 4, 2, 1),
            layers.LeakyReLU(0.2),
            # 128 x 16 x 16
            layers.Conv2D(128, 256, 4, 2, 1),
            layers.LeakyReLU(0.2),
            # Option 1: 256 x 8 x 8
            layers.Conv2D(256, 256, 4, 2, 1),
            layers.LeakyReLU(0.2),
            # 256 x 4 x 4
        )
        self.out = layers.Dense(256 * 4 * 4, z_dim)

    # @tf.function
    def call(self, x, **kwargs):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x


class LocallyLinearPredictor(tf.keras.layers.Layer):

    def __init__(self, z_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim

        my_layers = []
        for h in hidden_sizes:
            my_layers.append(layers.Dense(h, activation="relu"))
        my_layers.append(layers.Dense(z_dim * z_dim, activation=None))

        self.model = Sequential(my_layers)

    # @tf.function
    def call(self, inputs, **kwargs):
        z = inputs['z']
        a = inputs['a']
        x = tf.concat((z, a), dim=-1)
        Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim)  # b x z_dim x z_dim
        return tf.bmm(Ws, z.unsqueeze(-1)).squeeze(-1)  # b x z_dim
