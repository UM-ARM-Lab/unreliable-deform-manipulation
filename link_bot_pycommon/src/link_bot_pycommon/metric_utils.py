import numpy as np


def make_row(metric_name, e):
    return [metric_name, np.min(e), np.max(e), np.mean(e), np.median(e), np.std(e)]