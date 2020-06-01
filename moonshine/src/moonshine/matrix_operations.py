import tensorflow as tf


def batch_outer_product(a, b):
    """
    :param a: [batch, n]
    :param b: [batch, m]
    :return: [batch, n, m]
    """
    return tf.einsum('bn,bm->bnm', a, b)
