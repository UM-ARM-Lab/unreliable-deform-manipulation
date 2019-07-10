import numpy as np
from keras.layers import Dense, Concatenate, Lambda, Activation, Reshape

from link_bot_models.components.bias_layer import BiasLayer
from link_bot_models.components.out_of_bounds_regularization import OutOfBoundsRegularizer
from link_bot_models.components.sdf_lookup import SDFLookup


def sdf_function_layer(sdf_shape, fc_layer_sizes, beta, sigmoid_scale, output_name=None):
    p = "sdf_function_"
    # FIXME: make this passed in some how
    sdf_extent_cheating = np.array([-2.5, 2.5, -2.5, 2.5])
    regularizer = OutOfBoundsRegularizer(sdf_extent_cheating, beta)

    dense_layers = []
    for fc_layer_size in fc_layer_sizes:
        dense_layers.append(Dense(fc_layer_size, activation='tanh'))
    sdf_input_layer = Dense(2, activation=None, activity_regularizer=regularizer)
    concat = Concatenate(name=p + 'concat')
    sdf_lookup = SDFLookup(sdf_shape)
    negate = Lambda(lambda x: -x, name=p + 'negate')
    # NOTE: at the moment the resolution is 5cm and  the rope is only 2cm wide which is not good
    bias_layer = BiasLayer()
    scale_logits = Lambda(lambda x: sigmoid_scale * x, name=p + 'scale')
    sigmoid = Activation('sigmoid', name=output_name)

    # we have to flatten everything in order to pass it around and I don't understand why
    flatten_sdf = Reshape(target_shape=[sdf_shape[0] * sdf_shape[1]])
    flatten_sdf_gradient = Reshape(target_shape=[sdf_shape[0] * sdf_shape[1] * 2])

    def forward(sdf, sdf_gradient, sdf_resolution, sdf_origin, rope_input):
        fc_h = rope_input
        for dense in dense_layers:
            fc_h = dense(fc_h)
        sdf_input = sdf_input_layer(fc_h)

        sdf_flat = flatten_sdf(sdf)
        sdf_gradient_flat = flatten_sdf_gradient(sdf_gradient)
        sdf_func_inputs = concat([sdf_flat, sdf_gradient_flat, sdf_resolution, sdf_origin, sdf_input])

        signed_distance = sdf_lookup(sdf_func_inputs)
        negative_signed_distance = negate(signed_distance)
        bias = bias_layer(negative_signed_distance)
        logits = scale_logits(bias)
        predictions = sigmoid(logits)

        return predictions

    return sdf_input_layer, sdf_lookup, forward
