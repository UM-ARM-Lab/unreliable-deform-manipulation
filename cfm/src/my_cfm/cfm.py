from typing import Dict, List

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.models import Sequential

from link_bot_data.link_bot_dataset_utils import add_next
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from shape_completion_training.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class CFMNetwork(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.scenario = scenario

        self.observation_key = self.hparams['obs_key']
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']
        self.dataset_states_description = self.hparams['dynamics_dataset_hparams']['states_description']
        self.dataset_actions_description = self.hparams['dynamics_dataset_hparams']['action_description']

        self.encoder = Encoder(z_dim=self.hparams['z_dim'])
        self.predictor = LocallyLinearPredictor(z_dim=self.hparams['z_dim'], hidden_sizes=self.hparams['fc_layer_sizes'])

    def address_nans(self, example):
        new_example = {k: v for k, v in example.items()}
        # use a default values to replace NaN
        observation = example[self.observation_key]
        observation = tf.clip_by_value(observation, [[[[0, 0, 0, 0]]]], [[[[255, 255, 255, 3]]]])
        new_example[self.observation_key] = observation
        return new_example

    def normalize_observation(self, example):
        new_example = {k: v for k, v in example.items()}
        observation = example[self.observation_key] / tf.constant([[[255.0, 255.0, 255.0, 3.0]]])
        spatial_mean = tf.reduce_mean(tf.reduce_mean(observation, axis=2, keepdims=True), axis=2, keepdims=True)
        new_example[self.observation_key] = observation - spatial_mean
        return new_example

    def preprocess_no_gradient(self, example):
        new_example = self.address_nans(example)
        new_example = self.normalize_observation(new_example)
        return new_example

    def make_pairs(self, example):
        observation = example[self.observation_key]
        observation_pos = example[add_next(self.observation_key)]
        return observation, observation_pos

    def get_positive_pairs(self, example):
        observation = example[self.observation_key]
        return observation[:, :-1], observation[:, 1:]

    @tf.function
    def call(self, example, training=False, **kwargs):
        observation, observation_pos = self.get_positive_pairs(example)
        a = tf.concat([example[k] for k in self.action_keys], axis=-1)

        # forward pass
        z, z_pos = self.encoder(observation), self.encoder(observation_pos)  # b x z_dim
        z_next = self.predictor({'z': z, 'a': a})

        return {
            'z': z,
            'z_pos': z_pos,
            'z_next': z_next,
        }

    def compute_loss(self, example, outputs):
        z = outputs['z']
        z_pos = outputs['z_pos']
        z_next = outputs['z_next']
        z_size = z.shape[-1]

        # TODO: collapse various batch dimensions
        z = tf.reshape(z, [-1, z_size])
        z_pos = tf.reshape(z_pos, [-1, z_size])
        z_next = tf.reshape(z_next, [-1, z_size])
        batch_size = z.shape[0]

        # loss
        # NOTE: z could be z_next here? probably doesn't matter
        tiled_z = tf.tile(z[tf.newaxis], [batch_size, 1, 1])
        tiled_z_next = tf.tile(z_next[:, tf.newaxis], [1, batch_size, 1])
        neg_dists = tf.math.reduce_sum(tf.math.square(tiled_z - tiled_z_next), axis=-1)

        # Subtracting a large positive values should make the loss for diagonal elements be 0
        # which means we don't want to separate the representation of z from z_next
        neg_dists_masked = neg_dists - tf.one_hot(tf.range(batch_size), batch_size, on_value=1e12)  # b x b+1

        pos_dists = tf.math.reduce_sum(tf.math.square(z_pos - z_next), axis=-1, keepdims=True)

        dists = tf.concat((neg_dists_masked, pos_dists), axis=1)  # b x b+1
        log_probabilities = tf.nn.log_softmax(logits=dists, axis=-1)  # b x b+1
        loss = -tf.reduce_mean(log_probabilities[:, -1])  # Get last column which is the true pos sample

        return {
            'loss': loss
        }


class CFMWrapper(BaseDynamicsFunction):

    @staticmethod
    def get_net_class():
        return CFMNetwork


class Encoder(tf.keras.Model):

    def __init__(self, z_dim):
        super().__init__()

        self.z_dim = z_dim
        self.model = Sequential([
            layers.Conv2D(filters=64, kernel_size=3),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=64, kernel_size=4, strides=2),
            layers.LeakyReLU(0.2),
            # 64 x 32 x 32
            layers.Conv2D(filters=64, kernel_size=3, strides=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters=128, kernel_size=4, strides=2),
            layers.LeakyReLU(0.2),
            # 128 x 16 x 16
            layers.Conv2D(filters=256, kernel_size=4, strides=2),
            layers.LeakyReLU(0.2),
            # Option 1: 256 x 8 x 8
            layers.Conv2D(filters=256, kernel_size=4, strides=2),
            layers.LeakyReLU(0.2),
            # 256 x 4 x 4
        ], name='encoder')
        self.out = layers.Dense(z_dim)

    @tf.function
    def call(self, x, **kwargs):
        x = self.model(x)
        # NOTE: [:-3] gets all but the last 3 dimensions, which are the H, W, and C of the tensor
        # doing this specifically allows x to have multiple "batch" dimensions,
        # which is useful to treating [batch, time, ...] as all just batch dimensions
        x = tf.reshape(x, x.shape.as_list()[:-3] + [-1])
        x = self.out(x)
        return x


class LocallyLinearPredictor(tf.keras.layers.Layer):

    def __init__(self,
                 z_dim: int,
                 hidden_sizes: List[int]):
        super().__init__()
        self.z_dim = z_dim

        my_layers = []
        for h in hidden_sizes:
            my_layers.append(layers.Dense(h, activation="relu"))
        my_layers.append(layers.Dense(z_dim * z_dim, activation=None))

        self.model = Sequential(my_layers)

    @tf.function
    def call(self, inputs, **kwargs):
        z = inputs['z']
        a = inputs['a']
        x = tf.concat((z, a), axis=-1)
        linear_dynamics_params = self.model(x)
        linear_dynamics_matrix = tf.reshape(linear_dynamics_params, x.shape.as_list()[:-1] + [self.z_dim, self.z_dim])
        z_pred = tf.squeeze(tf.linalg.matmul(linear_dynamics_matrix, tf.expand_dims(z, axis=-1)), axis=-1)
        return z_pred
