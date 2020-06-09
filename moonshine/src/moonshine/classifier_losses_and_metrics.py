import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from link_bot_data.link_bot_dataset_utils import is_reconverging
from shape_completion_training import metric


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

    precision = metric.precision(y_true=valid_labels, y_pred=valid_probabilities)
    average_precision = tf.reduce_mean(precision)

    recall = metric.recall(y_true=valid_labels, y_pred=valid_probabilities)
    average_recall = tf.reduce_mean(recall)

    negative_accuracy = metric.accuray_on_negatives(y_true=valid_labels, y_pred=valid_probabilities)
    average_negative_accuracy = tf.reduce_mean(negative_accuracy)
    return {
        'accuracy': average_accuracy,
        'negative_accuracy': average_negative_accuracy,
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


def mdn_sequence_likelihood(dataset_element, predictions):
    return gaussian_negative_log_likelihood(y=dataset_element['action'],
                                            alpha=predictions['component_weights'],
                                            mu=predictions['means'],
                                            sigma=predictions['covariances'],
                                            mask=dataset_element['mask'])


def gaussian_negative_log_likelihood(y, alpha, mu, sigma, mask):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters. """
    scale_tril = tfp.math.fill_triangular(sigma)
    gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=tf.squeeze(alpha, 3)),
                               components_distribution=tfd.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril))
    log_likelihood = gm.log_prob(y)  # Evaluate log-probability of y
    valid_indices = tf.where(mask)
    valid_log_likelihood = tf.gather_nd(log_likelihood, valid_indices)

    return -tf.reduce_mean(valid_log_likelihood)
