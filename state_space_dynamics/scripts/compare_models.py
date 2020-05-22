#!/usr/bin/env python
import argparse
import json
import pathlib
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

from link_bot_data.dynamics_dataset import DynamicsDataset
from state_space_dynamics import model_utils
from link_bot_pycommon.args import my_formatter
from moonshine.gpu_config import limit_gpu_mem
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction

limit_gpu_mem(2)


def load_dataset_and_models(args):
    comparison_info = json.load(args.comparison.open("r"))
    models = {}
    for name, model_info in comparison_info.items():
        model_dir = model_info['model_dir']
        model, _ = model_utils.load_generic_model(model_dir)
        models[name] = model

    dataset = DynamicsDataset(args.dataset_dirs)
    tf_dataset = dataset.get_datasets(mode=args.mode,
                                      sequence_length=args.sequence_length,
                                      shard=args.shard,
                                      take=args.take).batch(1)

    if args.shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=1024)
    return tf_dataset, models


def run(args, base_folder: pathlib.Path, models: Dict[str, BaseDynamicsFunction], tf_dataset):
    all_metrics = {}
    for model_name, model in models.items():
        all_metrics[model_name] = {}
        metrics_for_model = {}

        print(Fore.GREEN + "Generating metrics for {}".format(model_name) + Fore.RESET)
        for example_idx, dataset_element in enumerate(tf_dataset):
            predictions = model.propagate_from_dataset_element(dataset_element)

            # Metrics
            metrics = model.scenario.dynamics_metrics_function(dataset_element, predictions)
            for metric_name, metric_value in metrics.items():
                if metric_name not in metrics_for_model:
                    metrics_for_model[metric_name] = []
                metrics_for_model[metric_name].append(metric_value.numpy())

            # Plotting
            if not args.no_plot or args.save:
                if example_idx < args.n_examples:
                    anim = model.scenario.animate_predictions_from_dynamics_dataset(example_idx, dataset_element, predictions)
                    if not args.no_plot:
                        plt.show()
                    if args.save:
                        # TODO: add animations to a list to save asynchronously? would need to keep all the figures around
                        filename = base_folder / '{}_anim_{}.gif'.format(model_name, example_idx)
                        from time import perf_counter
                        t0 = perf_counter()
                        anim.save(filename, writer='imagemagick', dpi=100)
                        write_dt = perf_counter() - t0
                        print('Saved animation {}/{} ({:.4f}s)'.format(example_idx + 1, args.n_examples, write_dt))

        for metric_name, metric_values in metrics_for_model.items():
            mean_metric_value = float(np.mean(metric_values))
            all_metrics[model_name][metric_name] = mean_metric_value
            print("{} {:.4f}".format(metric_name, mean_metric_value))

    # Save the results
    results_filename = base_folder / 'metrics.json'
    print(Fore.GREEN + "Saving results to {}".format(results_filename) + Fore.RESET)
    json.dump(all_metrics, results_filename.open("w"))


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+', help='dataset dirs')
    parser.add_argument('comparison', type=pathlib.Path, help='json file describing what should be compared')
    parser.add_argument('nickname', help='used to name output directory')
    parser.add_argument('--sequence-length', type=int, default=10, help='seq length')
    parser.add_argument('--no-plot', action='store_true', help='no plot')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test', help='mode')
    parser.add_argument('--n-examples', type=int, default=10, help='number of examples to visualize')
    parser.add_argument('--take', type=int, help='take only a subset of the data')
    parser.add_argument('--shard', type=int, help='shard only a subset of the data')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    plt.style.use("slides")

    base_folder = pathlib.Path('results') / 'compare_models' / '{}-{}-{}'.format(args.nickname, args.mode, int(time.time()))
    base_folder.mkdir(parents=True)
    print("Using output directory: {}".format(base_folder))

    tf_dataset, models = load_dataset_and_models(args)

    run(args, base_folder, models, tf_dataset)


if __name__ == '__main__':
    main()
