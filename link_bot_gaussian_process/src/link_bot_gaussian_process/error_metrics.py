import numpy as np

from link_bot_pycommon.src.link_bot_pycommon import link_bot_pycommon as tpoc


def make_row(metric_name, e):
    return [metric_name, np.min(e), np.max(e), np.mean(e), np.median(e), np.std(e)]


def fwd_model_error_metrics(my_model, test_x, test_y):
    """
    compute the euclidean distance for each node in pred_y[i] to each node in test_y[i],
    averaged over all i using the max likelihood prediction
    """
    pred_delta_x_mean, pred_delta_x_sigma = my_model.model.predict_y(test_x)
    tail_error = np.linalg.norm(pred_delta_x_mean[:, 0:2] - test_y[:, 0:2], axis=1)
    mid_error = np.linalg.norm(pred_delta_x_mean[:, 2:4] - test_y[:, 2:4], axis=1)
    head_error = np.linalg.norm(pred_delta_x_mean[:, 4:6] - test_y[:, 4:6], axis=1)
    total_node_error = tail_error + mid_error + head_error
    # each column goes [metric name, min, max, mean, median, std]
    return np.array([make_row('tail position error (m)', tail_error),
                     make_row('mid position error (m)', mid_error),
                     make_row('head position error (m)', head_error),
                     make_row('total position error (m)', total_node_error)], dtype=np.object)


def inv_model_error_metrics(my_model, test_x, test_y):
    """ compute the euclidean distance between the predicted control and the true control"""
    pred_y, _ = my_model.model.predict_y(test_x)
    pred_speeds = abs(pred_y[:, 2])

    abs_speed_error = abs(pred_speeds - abs(test_y[:, 2]))

    # compute dot product of each column of a with each column of b
    pred_theta = np.arctan2(pred_y[:, 1], pred_y[:, 0])
    true_theta = np.arctan2(test_y[:, 1], test_y[:, 0])
    abs_angle_error = abs(np.rad2deg(tpoc.yaw_diff(true_theta, pred_theta)))

    abs_time_step_error = abs(pred_y[:, 3] - test_y[:, 3])

    return np.array([make_row('speed error (m/s)', abs_speed_error),
                     make_row('angle error (deg)', abs_angle_error),
                     make_row('time steps error', abs_time_step_error)], dtype=np.object)
