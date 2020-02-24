#!/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import pathlib

import numpy as np
import tensorflow as tf

from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_pycommon.args import my_formatter

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results', type=pathlib.Path, help='a specific subfolder of a compare_planner results')
    # NOTE: these ought to match those used in collect_classifier_dataset.py
    parser.add_argument('--pre', type=float, help='pre threshold', default=0.15)
    parser.add_argument('--post', type=float, help='post threshold', default=0.21)
    parser.add_argument('--discard-pre-far', action='store_true', help='if true, dont count transitions where pre is far')
    parser.add_argument('--no-plot', action='store_true')

    args = parser.parse_args()
    filename = args.results / 'metrics.json'
    plans_metrics = json.load(filename.open("r"))["metrics"]

    post_far_count = 0
    post_close_pre_far_count = 0
    both_close_count = 0
    last_ts = []
    total_errors = []
    n_discarded = 0
    total_considered = 0
    became_close_again = 0
    for i, plan_metrics in enumerate(plans_metrics):
        actual_path = np.array(plan_metrics['actual_path'])
        planned_path = np.array(plan_metrics['planned_path'])

        t = 0
        was_pre_far = False
        for t in range(planned_path.shape[0] - 1):
            pre_planned_s = planned_path[t]
            pre_actual_s = actual_path[t]
            post_planned_s = planned_path[t + 1]
            post_actual_s = actual_path[t + 1]

            pre_transition_distance = np.linalg.norm(pre_planned_s - pre_actual_s)
            post_transition_distance = np.linalg.norm(post_planned_s - post_actual_s)
            total_error = pre_transition_distance + post_transition_distance

            pre_close = pre_transition_distance < args.pre
            post_close = post_transition_distance < args.post

            if not args.no_plot:
                cx = pre_planned_s[-2]
                cy = pre_planned_s[-1]
                extent = [cx - 0.25, cx + 0.25, cy - 0.25, cy + 0.25, ]
                plot_classifier_data(planned_state=pre_planned_s,
                                     planned_next_state=post_planned_s,
                                     state=pre_actual_s,
                                     next_state=post_actual_s,
                                     planned_env_extent=extent,
                                     label=post_close)
                plt.legend()
                plt.show()

            if not pre_close:
                was_pre_far = True
                if args.discard_pre_far:
                    total_considered += t
                    n_discarded += (planned_path.shape[0] - t)
                    break
                else:
                    total_considered += 1
            elif was_pre_far:
                was_pre_far = False
                became_close_again += 1

            if post_close and not pre_close:
                post_close_pre_far_count += 1
            if post_close and pre_close:
                both_close_count += 1
            if not post_close:
                post_far_count += 1

            total_errors.append(total_error)
        if args.discard_pre_far:
            last_ts.append(t)

    percent = post_far_count / total_considered * 100
    print("Post-Far Count: {} / {}  ({:4.2f}%)".format(post_far_count, total_considered, percent))
    print("Mean Error: {:5.3f}".format(np.mean(total_errors)))
    print("Median Error: {:5.3f}".format(np.median(total_errors)))
    print("Became close again {}".format(became_close_again))
    print("Both Close Count: {} / {}".format(both_close_count, total_considered))
    if args.discard_pre_far:
        print("Post-Close but Pre-Far Count: {} / {}".format(post_close_pre_far_count, total_considered))
        print("Discarded: {}".format(n_discarded))
        print(np.median(last_ts), np.mean(last_ts), np.std(last_ts))


if __name__ == '__main__':
    main()
