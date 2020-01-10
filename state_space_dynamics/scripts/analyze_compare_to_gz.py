#!/usr/bin/env python
import argparse
import pathlib
import matplotlib.pyplot as plt

import numpy as np
from tabulate import tabulate

from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.metric_utils import make_row


def error_metrics(head_error, mid_error, tail_error):
    return np.array([
        make_row('overall head error (m)', head_error),
        make_row('overall mid error (m)', mid_error),
        make_row('overall tail error (m)', tail_error),
        make_row('final head error (m)', head_error[:, -1]),
        make_row('final mid error (m)', mid_error[:, -1]),
        make_row('final tail error (m)', tail_error[:, -1]),
        make_row('final total error (m)', head_error[:, -1] + mid_error[:, -1] + tail_error[:, -1]),
    ], dtype=np.object)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dir', type=pathlib.Path, help='directory containing metrcs.npz file')

    args = parser.parse_args()

    metrics_filename = args.results_dir / 'metrics.npz'
    data = np.load(metrics_filename)

    head_errors = data['error'][:, :, 2]
    mid_errors = data['error'][:, :, 1]
    tail_errors = data['error'][:, :, 0]

    headers = ['error metric', 'min', 'max', 'mean', 'median', 'std']
    aggregate_metrics = error_metrics(head_error=head_errors, mid_error=mid_errors, tail_error=tail_errors)
    table = tabulate(aggregate_metrics, headers=headers, tablefmt='github', floatfmt='6.4f')
    print(table)

    mean_errors = np.mean(tail_errors + head_errors + mid_errors, axis=1)
    plt.figure()
    plt.scatter(data['initial_angle'], mean_errors)
    plt.plot([0, np.pi], [0, 0], c='k')
    plt.xlabel("angle (rad)")
    plt.ylabel("increase in prediction error in R^{n_state} (m)")
    plt.show()


if __name__ == '__main__':
    main()
