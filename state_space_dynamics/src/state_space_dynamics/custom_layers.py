import tensorflow as tf
from tensorflow.keras.layers import Lambda


def squeeze_layer(x, squeeze_dims, name='squeeze'):
    squeezed = Lambda(lambda x: tf.squeeze(x, squeeze_dims=squeeze_dims), name=name)(x)
    return squeezed
