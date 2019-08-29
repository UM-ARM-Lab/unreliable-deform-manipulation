import tensorflow as tf
from keras.layers import Reshape, Lambda


def action_smear_layer(action, conv_h):
    action_dim = int(action.shape[1])
    h = int(conv_h.shape[1])
    w = int(conv_h.shape[2])

    reshape = Reshape(target_shape=[1, 1, action_dim])
    smear = Lambda(function=lambda action_reshaped: tf.tile(action_reshaped, [1, h, w, 1]))

    def forward(action):
        action_reshaped = reshape(action)
        action_smear = smear(action_reshaped)
        return action_smear

    return forward
