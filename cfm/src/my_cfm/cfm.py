from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.models import Sequential

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.image_augmentation import augment
from moonshine.loss_utils import loss_on_dicts
from moonshine.moonshine_utils import vector_to_dict
from shape_completion_training.my_keras_model import MyKerasModel
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.base_filter_function import BaseFilterFunction


class CFM(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.scenario = scenario

        self.obs_keys = self.hparams['obs_keys']
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']
        self.observation_feature_keys = self.hparams['observation_feature_keys']
        self.image_key = 'color_depth_image'
        self.use_observation_feature_loss = self.hparams['use_observation_feature_loss']

        self.encoder = Encoder(hparams, batch_size, scenario)
        self.dynamics = LocallyLinearPredictor(hparams, batch_size, scenario)
        self.observer = Observer(hparams, batch_size, scenario)

        # check for NaNs, because the Kinect uses NaN to indicate "missing" data (usually out of range)
        # tf.debugging.enable_check_numerics()

    def apply_gradients(self, tape, train_element, train_outputs, losses):
        train_batch_loss = losses['loss']
        variables = self.trainable_variables
        gradients = tape.gradient(train_batch_loss, variables)
        # g can be None if there are parts of the network not being trained, i.e. the observer with there are no obs. feats.
        gradients = [tf.clip_by_norm(g, 1) for g in gradients if g is not None]
        self.optimizer.apply_gradients(zip(gradients, variables))
        return {}

    def normalize(self, example: Dict):
        # this nonsense is required otherwise calling this inside @tf.function complains about modifying python arguments
        new_example = {}
        for k, v in example.items():
            # Rescale to -1 to 1
            if k in self.obs_keys:
                min_value = tf.math.reduce_min(tf.math.reduce_min(example[k], axis=2, keepdims=True), axis=3, keepdims=True)
                max_value = tf.math.reduce_max(tf.math.reduce_max(example[k], axis=2, keepdims=True), axis=3, keepdims=True)
                normalized_observation = 2 * (example[k] - min_value) / (max_value - min_value) - 1
                new_example[k] = normalized_observation
            elif k in self.action_keys:
                min_value = tf.math.reduce_min(example[k], axis=2, keepdims=True)
                max_value = tf.math.reduce_max(example[k], axis=2, keepdims=True)
                normalized_action = 2 * (example[k] - min_value) / (max_value - min_value) - 1
                new_example[k] = normalized_action
            else:
                new_example[k] = v

        return new_example

    def get_positive_pairs(self, example):
        obs = {k: example[k][:, :-1] for k in self.obs_keys}
        obs_pos = {k: example[k][:, 1:] for k in self.obs_keys}
        return obs, obs_pos

    def preprocess_no_gradient(self, example):
        example = self.normalize(example)
        augmented = augment(example[self.image_key],
                            image_h=self.scenario.IMAGE_H,
                            image_w=self.scenario.IMAGE_W)
        example[self.image_key] = augmented
        return example

    # @tf.function
    def call(self, example, training=False, **kwargs):
        # NOTE: we are right now only doing 1-step prediction, and the perception predictions are 0-step
        observation, observation_pos = self.get_positive_pairs(example)

        # forward pass
        z, z_pos = self.encoder(observation), self.encoder(observation_pos)  # b x z_dim
        pred_inputs = {
            'z': z['z'],
            'z_pos': z_pos['z'],
        }
        for k in self.action_keys:
            pred_inputs[k] = example[k]
        z_seq = self.dynamics(pred_inputs)
        y_seq = self.observer(pred_inputs)

        output = {
            'z_pos': z_pos['z'],
        }
        output.update(z_seq)
        output.update(y_seq)
        return output

    def compute_loss(self, example, outputs):
        z = outputs['z'][:, 0]
        z_pos = outputs['z_pos'][:, 0]
        z_next = outputs['z'][:, 1]  # this assumes single transitions

        batch_size = z.shape[0]

        cfm_loss = self.cfm_loss(batch_size, z, z_next, z_pos)
        observation_feature_loss = self.observer_loss(example, outputs)
        loss = cfm_loss
        if self.use_observation_feature_loss:
            loss = loss + observation_feature_loss

        return {
            'loss': loss
        }

    def observer_loss(self, example, outputs):
        observation_feature_true = {k: example[k][:, :-1] for k in self.observation_feature_keys}
        observation_feature_predictions = {k: outputs[k] for k in self.observation_feature_keys}
        observation_feature_loss = loss_on_dicts(tf.keras.losses.mse,
                                                 dict_true=observation_feature_true,
                                                 dict_pred=observation_feature_predictions)
        return observation_feature_loss

    def cfm_loss(self, batch_size, z, z_next, z_pos):
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
        return loss


class Encoder(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']
        self.obs_keys = self.hparams['obs_keys']

        self.z_dim = self.hparams['z_dim']
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
        self.out = layers.Dense(self.z_dim)

    # @tf.function
    def call(self, observation: Dict, **kwargs):
        o = tf.concat([observation[k] for k in self.obs_keys], axis=-1)
        h = self.model(o)
        # NOTE: [:-3] gets all but the last 3 dimensions, which are the H, W, and C of the tensor
        # doing this specifically allows x to have multiple "batch" dimensions,
        # which is useful to treating [batch, time, ...] as all just batch dimensions
        h = tf.reshape(h, h.shape.as_list()[:-3] + [-1])
        z = self.out(h)
        return {
            'z': z
        }


class CFMFilter(BaseFilterFunction):

    @staticmethod
    def get_net_class():
        return Encoder


class LocallyLinearPredictor(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.observation_key = self.hparams['obs_keys']
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']

        self.z_dim = self.hparams['z_dim']

        my_layers = []
        for h in self.hparams['fc_layer_sizes']:
            my_layers.append(layers.Dense(h, activation="relu"))
        my_layers.append(layers.Dense(self.z_dim * self.z_dim, activation=None))

        self.model = Sequential(my_layers)

    # @tf.function
    def call(self, inputs, **kwargs):
        a = tf.concat([inputs[k] for k in self.action_keys], axis=-1)

        z = inputs['z']
        x = tf.concat((z, a), axis=-1)
        linear_dynamics_params = self.model(x)
        linear_dynamics_matrix = tf.reshape(linear_dynamics_params, x.shape.as_list()[:-1] + [self.z_dim, self.z_dim])
        z_next = tf.squeeze(tf.linalg.matmul(linear_dynamics_matrix, tf.expand_dims(z, axis=-1)), axis=-1)
        z_seq = tf.concat([z, z_next], axis=1)
        return {
            'z': z_seq
        }

    def compute_loss(self, dataset_element, outputs):
        raise NotImplementedError()


class Observer(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__(hparams=hparams, batch_size=batch_size)
        self.state_keys = self.hparams['state_keys']
        self.observation_features_keys = self.hparams['observation_feature_keys']
        dataset_params = self.hparams['dynamics_dataset_hparams']
        self.original_obs_feat_desc: Dict = dataset_params['observation_features_description']
        self.observation_features_description = {k: self.original_obs_feat_desc[k] for k in self.observation_features_keys}
        final_dim = sum(list(self.observation_features_description.values()))

        my_layers = []
        for h in self.hparams['fc_layer_sizes']:
            my_layers.append(layers.Dense(h, activation="relu"))
        my_layers.append(layers.Dense(final_dim, activation=None))

        self.model = Sequential(my_layers)

    # @tf.function
    def call(self, inputs, **kwargs):
        z = tf.concat([inputs[k] for k in self.state_keys], axis=-1)
        y = self.model(z)
        y_dict = vector_to_dict(self.observation_features_description, y)
        return y_dict

    def compute_loss(self, dataset_element, outputs):
        raise NotImplementedError()


class CFMLatentDynamics(BaseDynamicsFunction):

    @staticmethod
    def get_net_class():
        return LocallyLinearPredictor
