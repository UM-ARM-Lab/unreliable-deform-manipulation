import tensorflow as tf


def loss_on_dicts(loss, dict_true, dict_pred):
    loss_by_key = []
    for k, y_true in dict_true.items():
        y_pred = dict_pred[k]
        l = loss(y_true=y_true, y_pred=y_pred)
        loss_by_key.append(l)
    return tf.reduce_mean(loss_by_key)