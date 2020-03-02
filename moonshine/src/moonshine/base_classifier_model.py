import tensorflow as tf


def binary_classification_loss_function(dataset_element, predictions):
    label = dataset_element['label']
    loss_function = tf.keras.losses.BinaryCrossentropy()
    return loss_function(y_true=label, y_pred=predictions)


def binary_classification_metrics_function(dataset_element, predictions):
    label = dataset_element['label']
    accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    accuracy_metric.update_state(y_true=label, y_pred=predictions)
    return {
        'accuracy': accuracy_metric.result()
    }
