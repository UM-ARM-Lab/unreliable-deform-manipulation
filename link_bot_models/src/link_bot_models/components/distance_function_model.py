import keras.backend as K
from keras.layers import Lambda, Activation, Conv2D, Layer

from link_bot_models.components.distance_matrix_layer import DistanceMatrix


# noinspection PyAttributeOutsideInit
class DistanceFunctionModel(Layer):

    def __init__(self, sigmoid_scale, **kwargs):
        super(DistanceFunctionModel, self).__init__(**kwargs)
        self.sigmoid_scale = sigmoid_scale

    def call(self, rope_input, **kwargs):
        distances = self.distance_matrix_layer(rope_input)
        weighted_distance = self.weighted_distance_layer(distances)
        logits = self.scale_logits(weighted_distance)

        # TODO: this model doesn't handle "or" like conditions on the distances, since it's doing a linear combination
        predictions = Activation('sigmoid')(logits)

        return predictions

    def build(self, input_shape):
        n_points = int(input_shape[1] / 2)
        sigmoid_scale = self.sigmoid_scale  # necessary due to how Lambdas are serialized

        # Define the layers used
        self.distance_matrix_layer = DistanceMatrix()
        self.weighted_distance_layer = Conv2D(1, (n_points, n_points), activation=None, use_bias=True)
        self.scale_logits = Lambda(lambda x: sigmoid_scale * K.squeeze(K.squeeze(x, 1), 1), name='scale')
        self.sigmoid = Activation('sigmoid')

        # Build layers
        self.distance_matrix_layer.build(input_shape)
        self.weighted_distance_layer.build([input_shape[0], n_points, n_points, 1])
        self.scale_logits.build([input_shape[0], 1])
        self.sigmoid.build([input_shape[0], 1])

        # Add their parameters
        self._trainable_weights.extend(self.distance_matrix_layer.trainable_weights)
        self._trainable_weights.extend(self.weighted_distance_layer.trainable_weights)
        self._trainable_weights.extend(self.scale_logits.trainable_weights)
        self._trainable_weights.extend(self.sigmoid.trainable_weights)
        super(DistanceFunctionModel, self).build(input_shape)

    def get_config(self):
        config = {
            'sigmoid_scale': self.sigmoid_scale,
        }
        config.update(super(DistanceFunctionModel, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
