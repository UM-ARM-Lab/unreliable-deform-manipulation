#!/usr/bin/env python
import argparse
import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style
from matplotlib.animation import FuncAnimation
from tabulate import tabulate

from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.metric_utils import row_stats, dict_to_pvalue_table
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import listify, numpify, dict_of_sequences_to_sequence_of_dicts, remove_batch
from state_space_dynamics import model_utils

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


def generate(args):
    base_folder = pathlib.Path('results') / 'compare_models' / '{}-{}-{}'.format(args.nickname, args.mode, int(time.time()))
    base_folder.mkdir(parents=True)
    print("Using output directory: {}".format(base_folder))

    tf_dataset, models = load_dataset_and_models(args)

    all_data = []
    for example_idx, dataset_element in enumerate(tf_dataset):
        data_per_model_for_element = {}
        print(f'running prediction for example {example_idx}')
        for model_name, model in models.items():
            predictions = model.propagate_from_dataset_element(dataset_element)
            data_per_model_for_element[model_name] = {
                'dataset_element': dataset_element,
                'predictions': predictions,
                'scenario': model.scenario.simple_name(),
            }
        all_data.append(data_per_model_for_element)
    results_filename = base_folder / 'saved_data.json'
    print(Fore.GREEN + "Saving results to {}".format(results_filename) + Fore.RESET)
    with results_filename.open("w") as results_file:
        json.dump(listify(all_data), results_file)
    viz(results_filename, args.fps, args.no_plot, args.save)


def viz_main(args):
    viz(args.data_filename, args.fps, args.no_plot, args.save)


def viz(data_filename, fps, no_plot, save):
    plt.style.use("slides")

    # Load the results
    base_folder = data_filename.parent
    with data_filename.open("r") as data_file:
        saved_data = json.load(data_file)

    all_metrics = {}
    for example_idx, datum in enumerate(saved_data):
        # Plotting
        fig = plt.figure()
        ax = plt.gca()
        update_funcs = []
        frames = None
        for model_name, data_for_model in datum.items():
            scenario = get_scenario(data_for_model['scenario'])
            dataset_element = (numpify(data_for_model['dataset_element'][0]), numpify(data_for_model['dataset_element'][1]))
            inputs = remove_batch(numpify(data_for_model['dataset_element'][0]))
            outputs = remove_batch(numpify(data_for_model['dataset_element'][1]))
            predictions = remove_batch(numpify(data_for_model['predictions']))
            actions = inputs['action']
            actual = dict_of_sequences_to_sequence_of_dicts(outputs)
            predictions = dict_of_sequences_to_sequence_of_dicts(predictions)
            extent = inputs['full_env/extent']
            environment = {
                'full_env/env': inputs['full_env/env'],
                'full_env/extent': extent,
            }

            # Metrics
            metrics_for_model = {}
            metrics = scenario.dynamics_metrics_function(dataset_element, numpify(data_for_model['predictions']))
            loss = scenario.dynamics_loss_function(dataset_element, numpify(data_for_model['predictions']))
            metrics['loss'] = loss
            for metric_name, metric_value in metrics.items():
                if metric_name not in metrics_for_model:
                    metrics_for_model[metric_name] = []
                metrics_for_model[metric_name].append(metric_value.numpy())

            for metric_name, metric_values in metrics_for_model.items():
                mean_metric_value = float(np.mean(metric_values))
                if model_name not in all_metrics:
                    all_metrics[model_name] = {}
                if metric_name not in all_metrics[model_name]:
                    all_metrics[model_name][metric_name] = []
                all_metrics[model_name][metric_name].append(mean_metric_value)

            # Plotting
            update, frames = scenario.animate_predictions_on_axes(ax=ax,
                                                                  fig=fig,
                                                                  environment=environment,
                                                                  actions=actions,
                                                                  actual=actual,
                                                                  predictions=predictions,
                                                                  example_idx=example_idx,
                                                                  prediction_label_name=model_name,
                                                                  prediction_color=None)
            update_funcs.append(update)

        def update(t):
            for update_func in update_funcs:
                update_func(t)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        anim = FuncAnimation(fig, update, interval=1000 / fps, repeat=True, frames=frames)
        if not no_plot:
            plt.show()
        if save:
            filename = base_folder / '{}_anim_{}.gif'.format(model_name, example_idx)
            print(f"Saving {filename.as_posix()}")
            anim.save(filename, writer='imagemagick', dpi=100)
        plt.close(fig)

    metrics_by_model = {}
    for model_name, metrics_for_model in all_metrics.items():
        for metric_name, metric_values in metrics_for_model.items():
            if metric_name not in metrics_by_model:
                metrics_by_model[metric_name] = {}
            metrics_by_model[metric_name][model_name] = metric_values

    with (base_folder / 'metrics_tables.txt').open("w") as metrics_file:
        for metric_name, metric_by_model in metrics_by_model.items():
            headers = ["Model", "min", "max", "mean", "median", "std"]
            table_data = []
            for model_name, metric_values in metric_by_model.items():
                table_data.append([model_name] + row_stats(metric_values))
            print('-' * 90)
            print(Style.BRIGHT + metric_name + Style.NORMAL)
            table = tabulate(table_data,
                             headers=headers,
                             tablefmt='fancy_grid',
                             floatfmt='6.4f',
                             numalign='center',
                             stralign='left')
            metrics_file.write(table)
            print(table)
            print()

            print(Style.BRIGHT + f"p-value matrix [{metric_name}]" + Style.NORMAL)
            print(dict_to_pvalue_table(metric_by_model))


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()
    gen_parser = subparsers.add_parser('generate')
    gen_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+', help='dataset dirs')
    gen_parser.add_argument('comparison', type=pathlib.Path, help='json file describing what should be compared')
    gen_parser.add_argument('nickname', help='used to name output directory')
    gen_parser.add_argument('--sequence-length', type=int, default=10, help='seq length')
    gen_parser.add_argument('--no-plot', action='store_true', help='no plot')
    gen_parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test', help='mode')
    gen_parser.add_argument('--take', type=int, default=10, help='number of examples to visualize')
    gen_parser.add_argument('--shard', type=int, help='shard only a subset of the data')
    gen_parser.add_argument('--shuffle', action='store_true')
    gen_parser.add_argument('--save', action='store_true')
    gen_parser.add_argument('--fps', type=float, default=1)
    gen_parser.set_defaults(func=generate)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('data_filename', type=pathlib.Path, help='saved data from generate')
    viz_parser.add_argument('--fps', type=float, default=1)
    viz_parser.add_argument('--save', action='store_true')
    viz_parser.add_argument('--no-plot', action='store_true')
    viz_parser.set_defaults(func=viz_main)

    args = parser.parse_args()
    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
