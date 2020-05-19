import tensorflow as tf


def binary_classification_sequence_loss_function(dataset_element, predictions):
    # skip the first element, the label will always be 1
    labels = tf.expand_dims(dataset_element['is_close'][:, 1:], axis=2)
    # automatically sums over batch & time
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    total_bce = bce(y_true=labels, y_pred=predictions)
    return total_bce


def binary_classification_sequence_metrics_function(dataset_element, predictions):
    labels = tf.expand_dims(dataset_element['is_close'][:, 1:], axis=2)
    accuracy_over_time = tf.keras.metrics.binary_accuracy(y_true=labels, y_pred=predictions)
    average_accuracy = tf.reduce_mean(accuracy_over_time)
    return {
        'accuracy': average_accuracy
    }


def binary_classification_loss_function(dataset_element, predictions):
    label = dataset_element['label']
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true=label, y_pred=predictions)


def binary_classification_metrics_function(dataset_element, predictions):
    label = dataset_element['label']
    accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    accuracy_metric.update_state(y_true=label, y_pred=predictions)
    return {
        'accuracy': accuracy_metric.result()
    }
