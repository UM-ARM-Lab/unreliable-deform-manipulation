import keras.backend as K
from keras.layers import Layer, Lambda, Activation, Conv2D

from link_bot_models.components.distance_matrix_layer import DistanceMatrix


class DistanceFunctionLayer(Layer):

    def __init__(self, sdf_shape, fc_layer_sizes, beta, sigmoid_scale, **kwargs):
        super(DistanceFunctionLayer, self).__init__(**kwargs)
        self.sdf_shape = sdf_shape
        self.fc_layer_sizes = fc_layer_sizes
        self.beta = beta
        self.sigmoid_scale = sigmoid_scale
        # TODO: figure out how to access this layer without storing it in self like this hack
        self.distance_matrix_layer = None

    def call(self, rope_input, **kwargs):
        n_points = 3
        distances = DistanceMatrix()(rope_input)
        self.distance_matrix_layer = distances
        conv = Conv2D(1, (n_points, n_points), activation=None, use_bias=True)
        z = conv(distances)
        sigmoid_scale = self.sigmoid_scale
        z = Lambda(lambda x: K.squeeze(x, 1), name='squeeze1')(z)
        logits = Lambda(lambda x: sigmoid_scale * K.squeeze(x, 1), name='squeeze2')(z)

        # TODO: this model doesn't handle "or" like conditions on the distances, since it's doing a linear combination
        predictions = Activation('sigmoid', name='combined_output')(logits)

        return predictions

    def get_config(self):
        config = {
            'sigmoid_scale': self.sigmoid_scale,
        }
        base_config = super(DistanceFunctionLayer, self).get_config()
        return base_config.update(config)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1
