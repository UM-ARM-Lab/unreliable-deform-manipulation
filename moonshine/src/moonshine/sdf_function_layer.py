import numpy as np
from tensorflow.keras import layers

from moonshine.bias_layer import BiasLayer
from moonshine.out_of_bounds_regularization import OutOfBoundsRegularizer
from moonshine.sdf_lookup import SDFLookup


def sdf_function_layer(sdf_shape, fc_layer_sizes, beta, sigmoid_scale, output_name=None):
    p = "sdf_function_"
    # FIXME: make this passed in some how
    sdf_extent_cheating = np.array([-2.5, 2.5, -2.5, 2.5])
    regularizer = OutOfBoundsRegularizer(sdf_extent_cheating, beta)

    dense_layers = []
    for fc_layer_size in fc_layer_sizes:
        dense_layers.append(layers.Dense(fc_layer_size, activation='relu'))
    sdf_input_layer = layers.Dense(2, activation=None, activity_regularizer=regularizer)
    concat = layers.Concatenate(name=p + 'concat')
    sdf_lookup = SDFLookup(sdf_shape)
    negate = layers.Lambda(lambda x: -x, name=p + 'negate')
    # NOTE: at the moment the resolution is 5cm and  the rope is only 2cm wide which is not good
    bias_layer = BiasLayer()
    scale_logits = layers.Lambda(lambda x: sigmoid_scale * x, name=p + 'scale')
    sigmoid = layers.Activation('sigmoid', name=output_name)

    # we have to flatten everything in order to pass it around and I don't understand why
    flatten_sdf = layers.Reshape(target_shape=[sdf_shape[0] * sdf_shape[1]])
    flatten_sdf_gradient = layers.Reshape(target_shape=[sdf_shape[0] * sdf_shape[1] * 2])

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
