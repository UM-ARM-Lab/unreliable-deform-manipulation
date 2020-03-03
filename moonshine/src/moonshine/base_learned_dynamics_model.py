import tensorflow as tf

from moonshine.loss_utils import loss_on_dicts
from link_bot_pycommon.link_bot_pycommon import print_dict


def dynamics_loss_function(dataset_element, predictions):
    loss = tf.keras.losses.MeanSquaredError()
    input_data, output_data = dataset_element
    return loss_on_dicts(loss, dict_true=output_data, dict_pred=predictions)


def dynamics_metrics_function(dataset_element, predictions):
    input_data, output_data = dataset_element
    metrics = {}
    for state_key, pred_state in predictions.items():
        true_state = output_data[state_key]
        pred_points = tf.reshape(pred_state, [pred_state.shape[0], pred_state.shape[1], -1, 2])
        true_points = tf.reshape(true_state, [true_state.shape[0], true_state.shape[1], -1, 2])
        batch_position_error = tf.reduce_mean(tf.linalg.norm(pred_points - true_points, axis=3))
        last_pred_point = pred_points[:, -1]
        last_true_point = true_points[:, -1]
        final_tail_position_error = tf.reduce_mean(tf.linalg.norm(last_pred_point - last_true_point, axis=2))

        metrics['{} full error'.format(state_key)] = batch_position_error
        metrics['{} final error'.format(state_key)] = final_tail_position_error
    return metrics
