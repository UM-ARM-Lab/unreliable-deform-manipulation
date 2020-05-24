import numpy as np


def make_row(metric_name, metric_data):
    row = [metric_name]
    row.extend(row_stats(metric_data))
    return row


def row_stats(metric_data):
    return [np.min(metric_data), np.max(metric_data), np.mean(metric_data), np.median(metric_data), np.std(metric_data)]


def brief_row_stats(metric_data):
    return [np.mean(metric_data), np.median(metric_data), np.std(metric_data)]
