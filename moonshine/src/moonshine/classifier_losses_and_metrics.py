import tensorflow as tf


def binary_classification_sequence_loss_function(dataset_element, predictions):
    # skip the first element, the label will always be 1
    labels = tf.expand_dims(dataset_element['is_close'][:, 1:], axis=2)
    logits = predictions['logits']
    mask_float = tf.cast(predictions['mask'][:, 1:], tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
    # mask to ignore loss for states
    bce = mask_float * bce
    # mean over batch & time
    total_bce = tf.reduce_mean(bce)
    return total_bce


def binary_classification_sequence_metrics_function(dataset_element, predictions):
    labels = tf.expand_dims(dataset_element['is_close'][:, 1:], axis=2)
    logits = predictions['logits']
    mask_float = tf.cast(predictions['mask'][:, 1:], tf.float32)
    accuracy_over_time = tf.keras.metrics.binary_accuracy(y_true=labels, y_pred=logits)
    accuracy_over_time = mask_float * accuracy_over_time
    average_accuracy = tf.reduce_mean(accuracy_over_time)
    return {
        'accuracy': average_accuracy
    }


def binary_classification_loss_function(dataset_element, predictions):
    label = dataset_element['label']
    # because RNN masking handles copying of hiddne states, the final logit is the same as the last "valid" logit
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
