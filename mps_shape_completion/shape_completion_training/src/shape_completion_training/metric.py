import tensorflow as tf


class Metric:

    @staticmethod
    def is_better_than(a, b):
        raise NotImplementedError()

    @staticmethod
    def key():
        raise NotImplementedError()

    @staticmethod
    def worst():
        raise NotImplementedError()


class LossMetric(Metric):

    @staticmethod
    def is_better_than(a, b):
        return a < b

    @staticmethod
    def key():
        return "loss"

    @staticmethod
    def worst():
        return 1000


class AccuracyMetric(Metric):

    @staticmethod
    def is_better_than(a, b):
        if b is None:
            return True
        return a > b

    @staticmethod
    def key():
        return "accuracy"

    @staticmethod
    def worst():
        return 0

# TODO make tests for these


def fp(y_true, y_pred, threshold=0.5):
    return tf.cast(tf.math.count_nonzero((1 - y_true) * tf.cast(y_pred > threshold, tf.float32)), tf.float32)


def tn(y_true, y_pred, threshold=0.5):
    return tf.cast(tf.math.count_nonzero((1 - y_true) * tf.cast(y_pred <= threshold, tf.float32)), tf.float32)


def fn(y_true, y_pred, threshold=0.5):
    return tf.cast(tf.math.count_nonzero(y_true * tf.cast(y_pred <= threshold, tf.float32)), tf.float32)


def tp(y_true, y_pred, threshold=0.5):
    return tf.cast(tf.math.count_nonzero(y_true * tf.cast(y_pred > threshold, tf.float32)), tf.float32)


def accuray_on_negatives(y_true, y_pred, threshold=0.5):
    true_negatives = tn(y_true, y_pred, threshold=threshold)
    false_positives = fp(y_true, y_pred, threshold=threshold)
    return tf.math.divide_no_nan(true_negatives, true_negatives + false_positives)


def recall(y_true, y_pred, threshold=0.5):
    true_positives = tp(y_true, y_pred, threshold=threshold)
    false_negatives = fn(y_true, y_pred, threshold=threshold)
    return tf.math.divide_no_nan(true_positives, true_positives + false_negatives)


def precision(y_true, y_pred, threshold=0.5):
    true_positives = tp(y_true, y_pred, threshold=threshold)
    false_positives = fp(y_true, y_pred, threshold=threshold)
    return tf.math.divide_no_nan(true_positives, true_positives + false_positives)
