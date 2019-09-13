import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from link_bot_classifiers.components.action_smear_layer import action_smear_layer
from link_bot_classifiers.components.raster_points_layer import RasterPoints
from link_bot_classifiers.components.simple_cnn_layer import simple_cnn_relu_layer


class LocallyLinearModel:

    def __init__(self, hparams):
        self.hparams = hparams

        self.sdf_shape = hparams['sdf_shape']
        self.conv_filters = hparams['conv_filters']
        self.fc_layer_sizes = hparams['fc_layer_sizes']

        sdf = layers.Input(name='sdf', shape=(self.sdf_shape[0], self.sdf_shape[1], 1))
        rope_config = layers.Input(name='rope_configurations', shape=(self.hparams['n_points'],))
        sdf_resolution = layers.Input(name='sdf_resolution', shape=(2,))
        sdf_origin = layers.Input(name='sdf_origin', shape=(2,))
        actions = layers.Input(name='actions', shape=(2,))

        binary_sdf = sdf > 0
        action_image = action_smear_layer(actions, binary_sdf)(actions)
        rope_image = RasterPoints(self.sdf_shape)([rope_config, sdf_resolution, sdf_origin])
        concat = layers.Concatenate(axis=-1)([binary_sdf, action_image, rope_image])
        out_h = simple_cnn_relu_layer(self.conv_filters, self.fc_layer_sizes)(concat)
        # right now we output N^2 + N*M elements for full A and B matrices
        elements_in_A = self.hparams['n_points'] * self.hparams['n_points']
        elements_in_B = self.hparams['n_control']
        num_elements_in_linear_model = elements_in_A + elements_in_B
        A_and_B_elements = layers.Dense(num_elements_in_linear_model, activation='sigmoid', name='constraints')(out_h)

        s_t = rope_config
        for t, action in actions:
            A_t_perterbation, B_t = tf.split(A_and_B_elements, [elements_in_A, elements_in_B])
            A_t_perterbation = layers.Reshape((self.hparams['n_points'], self.hparams['n_points']))(A_t_perterbation)
            A_t = tf.eye(self.hparams['n_points']) + A_t_perterbation
            B_t = layers.Reshape((self.hparams['n_points'], self.hparams['n_control']))(B_t)
            u_t = action
            s_t_plus_1 = A_t @ s_t + B_t @ u_t

        self.model_inputs = [sdf, sdf_resolution, sdf_origin, action, rope_config]
        self.keras_model = models.Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam',
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'])

    def get_default_hparams(self):
        return {
        }

    @staticmethod
    def load(self, checkpoint_directory):
        pass

    def save(self, checkpoint_directory):
        pass
