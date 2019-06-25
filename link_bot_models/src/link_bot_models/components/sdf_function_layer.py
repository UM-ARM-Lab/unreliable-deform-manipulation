import numpy as np
from keras import Model
from keras.layers import Dense, Concatenate, Lambda, Activation

from link_bot_models.components.bias_layer import BiasLayer
from link_bot_models.components.out_of_bounds_regularization import OutOfBoundsRegularizer
from link_bot_models.components.sdf_lookup import SDFLookup


class SDFFunctionLayer(Model):

    def __init__(self, sdf_shape, fc_layer_sizes, beta, sigmoid_scale, **kwargs):
        super(SDFFunctionLayer, self).__init__(**kwargs)
        self.sdf_shape = sdf_shape
        self.fc_layer_sizes = fc_layer_sizes
        self.beta = beta
        self.sigmoid_scale = sigmoid_scale
        # TODO: figure out how to access this layer without storing it in self like this hack
        self.sdf_input_layer = None

    def call(self, inputs, **kwargs):
        sdf_flat = inputs[0]
        sdf_gradient_flat = inputs[1]
        sdf_resolution = inputs[2]
        sdf_origin = inputs[3]
        rope_input = inputs[4]

        fc_h = rope_input
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='tanh')(fc_h)
        # FIXME: sdf extent is a tensor which keras won't serialize correctly, and so the model
        # can't save/load correctly. Not sure how to work around this so I'm just hard coding this for now
        sdf_extent_cheating = np.array([-2.5, 2.5, -2.5, 2.5])
        regularizer = OutOfBoundsRegularizer(sdf_extent_cheating, self.beta)
        self.sdf_input_layer = Dense(2, activation=None, activity_regularizer=regularizer)
        sdf_input = self.sdf_input_layer(fc_h)

        sdf_func_inputs = Concatenate()([sdf_flat, sdf_gradient_flat, sdf_resolution, sdf_origin, sdf_input])

        signed_distance = SDFLookup(self.sdf_shape)(sdf_func_inputs)
        negative_signed_distance = Lambda(lambda x: -x)(signed_distance)
        bias = BiasLayer()(negative_signed_distance)
        logits = Lambda(lambda x: self.sigmoid_scale * x)(bias)
        # threshold = 0.0
        # logits = Lambda(lambda x: threshold - x)(signed_distance)
        predictions = Activation('sigmoid', name='combined_output')(logits)

        return predictions

    def get_config(self):
        config = {
            'fc_layer_sizes': self.fc_layer_sizes,
            'beta': self.beta,
            'sigmoid_scale': self.sigmoid_scale,
        }
        base_config = super(SDFFunctionLayer, self).get_config()
        return base_config.update(config)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1
