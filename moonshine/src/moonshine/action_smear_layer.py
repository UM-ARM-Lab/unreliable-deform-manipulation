import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


def action_smear_layer(input_sequence_length, action_dim, h, w):
    reshape = layers.Reshape(target_shape=[input_sequence_length, 1, 1, action_dim], name='smear_reshape')
    smear = layers.Lambda(function=lambda action_reshaped: tf.tile(action_reshaped, [1, 1, h, w, 1]),
                          name='spatial_tile')

    def forward(action):
        action_reshaped = reshape(action)
        action_smear = smear(action_reshaped)
        return action_smear

    return forward


def smear_action_differentiable(action, h, w):
    """
    :param action: [batch, n_action]
    :param h: scalar , int
    :param w: scalar, int
    :return:  [batch, h, w, n_action]
    """
    action_reshaped = tf.expand_dims(tf.expand_dims(action, axis=1), axis=1)
    action_image = tf.tile(action_reshaped, [1, h, w, 1])
    return action_image
