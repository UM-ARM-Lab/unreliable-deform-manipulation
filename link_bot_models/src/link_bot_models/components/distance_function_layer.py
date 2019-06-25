import keras.backend as K
from keras import Model
from keras.layers import Lambda, Activation, Conv2D

from link_bot_models.components.distance_matrix_layer import DistanceMatrix


class DistanceFunctionLayer(Model):

    def __init__(self, sigmoid_scale, **kwargs):
        super(DistanceFunctionLayer, self).__init__(**kwargs)
        self.sigmoid_scale = sigmoid_scale
        # TODO: figure out how to access this layer without storing it in self like this hack
        self.distance_matrix_layer = None
        self.conv = None

    def call(self, rope_input, **kwargs):
        n_points = 3
        distances_layer = DistanceMatrix()
        distances = distances_layer(rope_input)
        self.distance_matrix_layer = distances_layer
        self.conv = Conv2D(1, (n_points, n_points), activation=None, use_bias=True)
        z = self.conv(distances)
        sigmoid_scale = self.sigmoid_scale  # necessary due to how Lambdas are serialized
        z = Lambda(lambda x: K.squeeze(x, 1), name='squeeze1')(z)
        logits = Lambda(lambda x: sigmoid_scale * K.squeeze(x, 1), name='squeeze2')(z)

        # TODO: this model doesn't handle "or" like conditions on the distances, since it's doing a linear combination
        predictions = Activation('sigmoid')(logits)

        return predictions

    def get_config(self):
        config = {
            'sigmoid_scale': self.sigmoid_scale,
        }
        base_config = super(DistanceFunctionLayer, self).get_config()
        return base_config.update(config)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
