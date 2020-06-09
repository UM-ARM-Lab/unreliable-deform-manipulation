import tensorflow as tf
from tensorflow.keras import layers


def relu_layers(fc_layer_sizes, use_bias=True):
    dense_layers = []
    for fc_layer_size in fc_layer_sizes:
        dense_layers.append(layers.Dense(fc_layer_size, activation='relu', use_bias=use_bias, name='relu_dense'))

    def forward(x):
        fc_h = x
        for dense in dense_layers:
            fc_h = dense(fc_h)
        output = fc_h
        return output

    return forward


def nnelu(input):
    """
    Computes the Non-Negative Exponential Linear Unit
    used when we require the output of a neuron to be strictly positive,
    such as for producing positive definite matrices
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))
