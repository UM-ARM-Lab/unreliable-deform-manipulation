import tensorflow as tf
from tensorflow.keras import layers


def action_smear_layer(action, conv_h):
    action_dim = int(action.shape[1])
    h = int(conv_h.shape[1])
    w = int(conv_h.shape[2])

    reshape = layers.Reshape(target_shape=[1, 1, action_dim])
    smear = layers.Lambda(function=lambda action_reshaped: tf.tile(action_reshaped, [1, h, w, 1]), name='spatial_tile')

    def forward(action):
        action_reshaped = reshape(action)
        action_smear = smear(action_reshaped)
        return action_smear

    return forward
