import numpy as np

from link_bot_gaussian_process import link_bot_gp
from link_bot_pycommon import link_bot_pycommon


def make_row(metric_name, e):
    return [metric_name, np.min(e), np.max(e), np.mean(e), np.median(e), np.std(e)]


def multistep_fwd_model_error_metrics(fwd_model, test_x, test_y):
    """
    compute the euclidean distance for each node in pred_y[i] to each node in test_y[i],
    averaged over all i using the max likelihood prediction
    """
    x0 = test_x['states'][0, 0]
    x0 = np.expand_dims(x0, 0)
    actions = test_x['actions'][0]
    true = test_y['output_states'][0]
    prediction, _ = link_bot_gp.predict(fwd_model, x0, actions)
    prediction = prediction.reshape([-1, 3, 2])
    true = true.reshape([-1, 3, 2])

    error = np.linalg.norm(prediction - true, axis=2)
    total_error = np.sum(error, axis=1)
    tail_error = error[:, 0]
    mid_error = error[:, 1]
    head_error = error[:, 2]

    # each column goes [metric name, min, max, mean, median, std]
    return np.array([make_row('tail error (m)', tail_error),
                     make_row('mid error (m)', mid_error),
                     make_row('head error (m)', head_error),
                     make_row('total error (m)', total_error),
                     ], dtype=np.object)


def fwd_model_error_metrics(my_model, test_x, test_y):
    """
    compute the euclidean distance for each node in pred_y[i] to each node in test_y[i],
    averaged over all i using the max likelihood prediction
    """
    tail_disp = np.linalg.norm(test_y[:, 0:2], axis=1)
    head_disp = np.linalg.norm(test_y[:, 4:6], axis=1)

    pred_delta_x_mean, pred_delta_x_sigma = my_model.model.predict_y(test_x)
    tail_error = np.linalg.norm(pred_delta_x_mean[:, 0:2] - test_y[:, 0:2], axis=1)
    mid_error = np.linalg.norm(pred_delta_x_mean[:, 2:4] - test_y[:, 2:4], axis=1)
    head_error = np.linalg.norm(pred_delta_x_mean[:, 4:6] - test_y[:, 4:6], axis=1)
    total_node_error = tail_error + mid_error + head_error
    # each column goes [metric name, min, max, mean, median, std]
    return np.array([make_row('tail position error (m)', tail_error),
                     make_row('mid position error (m)', mid_error),
                     make_row('head position error (m)', head_error),
                     make_row('total position error (m)', total_node_error),
                     make_row("tail position displacement (m)", tail_disp),
                     make_row("head position displacement (m)", head_disp),
                     ], dtype=np.object)


def inv_model_error_metrics(my_model, test_x, test_y):
    """ compute the euclidean distance between the predicted control and the true control"""
    pred_y, _ = my_model.model.predict_y(test_x)

    # compute dot product of each column of a with each column of b
    pred_theta = np.arctan2(pred_y[:, 1], pred_y[:, 0])
    true_theta = np.arctan2(test_y[:, 1], test_y[:, 0])
    abs_angle_error = abs(np.rad2deg(link_bot_pycommon.yaw_diff(true_theta, pred_theta)))

    # pred_speeds = abs(pred_y[:, 2])
    # abs_speed_error = abs(pred_speeds - abs(test_y[:, 2]))

    # abs_time_step_error = abs(pred_y[:, 3] - test_y[:, 3])

    return np.array([
        make_row('angle error (deg)', abs_angle_error),
        # make_row('speed error (m/s)', abs_speed_error),
        # make_row('time steps error', abs_time_step_error)
    ], dtype=np.object)
