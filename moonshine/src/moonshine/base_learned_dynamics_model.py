import tensorflow as tf

from moonshine.loss_utils import loss_on_dicts


def dynamics_loss_function(dataset_element, predictions):
    loss = tf.keras.losses.MeanSquaredError()
    input_data, output_data = dataset_element
    return loss_on_dicts(loss, dict_true=output_data, dict_pred=predictions)


def dynamics_points_metrics_function(dataset_element, predictions):
    input_data, output_data = dataset_element
    metrics = {}
    for state_key, pred_state in predictions.items():
        true_state = output_data[state_key]
        pred_points = tf.reshape(pred_state, [pred_state.shape[0], pred_state.shape[1], -1, 2])
        true_points = tf.reshape(true_state, [true_state.shape[0], true_state.shape[1], -1, 2])
        all_distances = tf.linalg.norm(pred_points - true_points, axis=3)
        batch_position_error = tf.reduce_mean(all_distances)
        final_position_error = tf.reduce_mean(all_distances[:, -1])

        metrics['{} full error'.format(state_key)] = batch_position_error
        metrics['{} final error'.format(state_key)] = final_position_error
        for point_idx in range(all_distances.shape[2]):
            final_position_error_pt = tf.reduce_mean(all_distances[:, -1, point_idx])
            metrics['{} final error point #{}'.format(state_key, point_idx)] = final_position_error_pt
    return metrics


@tf.function
def ensemble_dynamics_loss_function(dataset_element, predictions):
    input_data, output_data = dataset_element
    # repeat the output data to match the number of ensembles
    n_ensembles = int(next(iter(predictions.values())).shape[1])
    output_data_repeated = dict([(k, tf.stack([v] * n_ensembles, axis=1)) for k, v in output_data.items()])

    loss = tf.keras.losses.MeanSquaredError()
    return loss_on_dicts(loss, dict_true=output_data_repeated, dict_pred=predictions)


@tf.function
def ensemble_dynamics_metrics_function(dataset_element, predictions):
    input_data, output_data = dataset_element
    metrics = {}
    for state_key, pred_state in predictions.items():
        mean_pred_state = tf.math.reduce_mean(pred_state, axis=1)
        true_state = output_data[state_key]
        pred_points = tf.reshape(mean_pred_state, [mean_pred_state.shape[0], mean_pred_state.shape[1], -1, 2])
        true_points = tf.reshape(true_state, [true_state.shape[0], true_state.shape[1], -1, 2])
        all_distances = tf.linalg.norm(pred_points - true_points, axis=3)
        batch_position_error = tf.reduce_mean(all_distances)
        final_position_error = tf.reduce_mean(all_distances[:, -1])

        metrics['{} full error'.format(state_key)] = batch_position_error
        metrics['{} final error'.format(state_key)] = final_position_error
        for point_idx in range(all_distances.shape[2]):
            final_position_error_pt = tf.reduce_mean(all_distances[:, -1, point_idx])
            metrics['{} final error point #{}'.format(state_key, point_idx)] = final_position_error_pt
    return metrics
