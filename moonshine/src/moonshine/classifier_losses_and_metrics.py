import tensorflow as tf

from link_bot_data.link_bot_dataset_utils import is_reconverging
from shape_completion_training.metric import recall, precision


def negative_weighted_binary_classification_sequence_loss_function(dataset_element, predictions):
    # skip the first element, the label will always be 1
    is_close = dataset_element['is_close'][:, 1:]
    labels = tf.expand_dims(is_close, axis=2)
    logits = predictions['logits']
    valid_indices = tf.where(predictions['mask'][:, 1:])
    bce = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
    # mask to ignore loss for states
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    total_bce = compute_weighted_mean_loss(bce, is_close, valid_indices)
    return total_bce


def compute_weighted_mean_loss(bce, positives, valid_indices):
    negatives = 1 - positives
    n_positive = tf.math.reduce_sum(positives)
    batch_size = positives.shape[0]
    n_negative = batch_size - n_positive
    # TODO: handle division by 0
    if tf.equal(n_positive, 0):
        weight_for_positive = tf.constant(100, dtype=tf.float32)
    else:
        weight_for_positive = batch_size / 2.0 / n_positive
    if tf.equal(n_negative, 0):
        weight_for_negative = tf.constant(100, dtype=tf.float32)
    else:
        weight_for_negative = batch_size / 2.0 / n_negative
    weighted_bce = tf.math.add(tf.math.multiply(bce, positives * weight_for_positive),
                               tf.math.multiply(bce, negatives * weight_for_negative))
    valid_weighted_bce = tf.gather_nd(weighted_bce, valid_indices)
    # mean over batch & time
    total_bce = tf.reduce_mean(valid_weighted_bce)
    return total_bce


def reconverging_weighted_binary_classification_sequence_loss_function(dataset_element, predictions):
    # skip the first element, the label will always be 1
    is_close = dataset_element['is_close'][:, 1:]
    labels = tf.expand_dims(is_close, axis=2)
    logits = predictions['logits']
    valid_indices = tf.where(predictions['mask'][:, 1:])
    bce = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
    # mask to ignore loss for states
    reconverging = tf.cast(is_reconverging(dataset_element['is_close']), tf.float32)
    T = is_close.shape[1]
    reconverging_per_step = tf.stack([reconverging] * T, axis=1)
    total_bce = compute_weighted_mean_loss(bce, reconverging_per_step, valid_indices)
    return total_bce


def binary_classification_sequence_loss_function(dataset_element, predictions):
    # skip the first element, the label will always be 1
    labels = tf.expand_dims(dataset_element['is_close'][:, 1:], axis=2)
    logits = predictions['logits']
    valid_indices = tf.where(predictions['mask'][:, 1:])
    bce = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
    # mask to ignore loss for states
    bce = tf.gather_nd(bce, valid_indices)
    # mean over batch & time
    total_bce = tf.reduce_mean(bce)
    return total_bce


def binary_classification_sequence_metrics_function(dataset_element, predictions):
    labels = tf.expand_dims(dataset_element['is_close'][:, 1:], axis=2)
    logits = predictions['logits']
    probabilities = predictions['probabilities']
    valid_indices = tf.where(predictions['mask'][:, 1:])
    valid_labels = tf.gather_nd(labels, valid_indices)
    valid_logits = tf.gather_nd(logits, valid_indices)
    valid_probabilities = tf.gather_nd(probabilities, valid_indices)
    accuracy_over_time = tf.keras.metrics.binary_accuracy(y_true=valid_labels, y_pred=valid_logits)
    average_accuracy = tf.reduce_mean(accuracy_over_time)

    precision_over_time = precision(y_true=valid_labels, y_pred=valid_probabilities)
    average_precision = tf.reduce_mean(precision_over_time)

    recall_over_time = recall(y_true=valid_labels, y_pred=valid_probabilities)
    average_recall = tf.reduce_mean(recall_over_time)
    return {
        'accuracy': average_accuracy,
        'precision': average_precision,
        'recall': average_recall,
    }


def binary_classification_loss_function(dataset_element, predictions):
    label = dataset_element['label']
    # because RNN masking handles copying of hidden states, the final logit is the same as the last "valid" logit
    logit = predictions['logits'][:, -1]
    bce = tf.keras.losses.binary_crossentropy(y_true=label, y_pred=logit, from_logits=True)
    # mean over batch & time
    total_bce = tf.reduce_mean(bce)
    return total_bce


def binary_classification_metrics_function(dataset_element, predictions):
    label = dataset_element['label']
    logit = predictions['logits'][:, -1]
    accuracy_over_time = tf.keras.metrics.binary_accuracy(y_true=label, y_pred=logit)
    average_accuracy = tf.reduce_mean(accuracy_over_time)
    return {
        'accuracy': average_accuracy
    }
