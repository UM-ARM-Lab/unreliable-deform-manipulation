#!/usr/bin/env python
import argparse
import numpy as np
import json
import pathlib
import pickle
import time
from typing import Dict

import matplotlib.pyplot as plt
import tensorflow as tf
from colorama import Fore

from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_planning import model_utils
from link_bot_planning.get_scenario import get_scenario
from link_bot_pycommon.args import my_formatter
from moonshine.numpy_utils import sequence_of_dicts_to_dict_of_sequences
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def generate(args):
    base_folder = pathlib.Path('results') / 'compare_models-{}-{}'.format(args.mode, int(time.time()))
    base_folder.mkdir(parents=True)

    tf_dataset, models = load_dataset_and_models(args)

    run(base_folder, models, tf_dataset)


def evaluate(args):
    results_filename = args.results_dir / 'results.pkl'
    metadata_filename = args.results_dir / 'metadata.json'
    results = pickle.load(results_filename.open("rb"))
    metadata = json.load(metadata_filename.open("r"))
    evaluate_metrics(results, metadata)


def visualize(args):
    results_filename = args.results_dir / 'results.pkl'
    metadata_filename = args.results_dir / 'metadata.json'
    results = pickle.load(results_filename.open("rb"))
    metadata = json.load(metadata_filename.open("r"))
    visualize_predictions(results, metadata, args.n_examples, args.results_dir)


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
                                      take=args.take).batch(1)
    return tf_dataset, models


def run(base_folder: pathlib.Path, models: Dict[str, BaseDynamicsFunction], tf_dataset):
    all_metrics = {}
    for model_name, model in models.items():
        metrics_for_model = {}
        print(Fore.CYAN + "Generating metrics for {}".format(model_name) + Fore.RESET)
        for dataset_element in tf_dataset:
            predictions = model.propagate_from_dataset_element(dataset_element)

            # Metrics
            metrics = model.scenario.dynamics_metrics_function(dataset_element, predictions)
            for metric_name, metric_value in metrics.items():
                if metric_name not in metrics_for_model:
                    metrics_for_model[metric_name] = []
                metrics_for_model[metric_name].append(metric_value.numpy())

            # Plotting
            model.scenario.plot_state()??? make a plot dataset element function?

        for metric_name, metric_values in metrics_for_model.items():
            mean_metric_value = np.mean(metric_values)
            all_metrics[model_name][metric_name] = mean_metric_value
            print(metric_name, mean_metric_value)

    # Save the results
    results_filename = base_folder / 'metrics.json'
    print(Fore.CYAN + "Saving results to {}".format(results_filename) + Fore.RESET)
    json.dump(all_metrics, results_filename.open("w"))

    # # Save some metadata
    # metadata_filename = base_folder / 'metadata.json'
    # metadata = []
    # for model_name, model in models.items():
    #     model_info = {
    #         'name': model_name,
    #         'hparams': model.hparams,
    #     }
    #     metadata.append(model_info)
    #
    # json.dump(metadata, metadata_filename.open('w'), indent=2)
    #
    # return results, metadata


def visualize_predictions(results, metadata, n_examples, results_dir):
    # results is a dictionary, where each key is the name of the model (or ground truth) and the value is a list of dictionaries
    # each element in the list is an example/trajectory, and each dictionary contains the state predicted by the model

    for example_idx in range(n_examples):
        plt.show()


def evaluate_metrics(results, metadata):
    for model_name in results.keys():
        if model_name == 'true':
            continue

        scenario = get_scenario(metadata[model_name])
        datset_element_like = sequence_of_dicts_to_dict_of_sequences(results['true'])
        metrics = []
        for predictions in results[model_name]:
            datset_element_like = sequence_of_dicts_to_dict_of_sequences(predictions)
            metric_i = scenario.dynamics_metrics_function(datset_element_like, predictions)
            metrics.append(metric_i)

        print()
        print("Model: {}".format(model_name))
        # for i in range(n_points):
        #     print("point {} error:  {:8.4f}m {:6.4f}".format(i, np.mean(errors_by_point[i]), np.std(errors_by_point[i])))
        # print("total error: {:8.4f}m {:6.4f}".format(np.mean(total_errors), np.std(total_errors)))
        # print("final_tail error: {:8.4f}m {:6.4f}".format(np.mean(final_tail_errors), np.std(final_tail_errors)))


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)

    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+', help='dataset dirs')
    generate_parser.add_argument('comparison', type=pathlib.Path, help='json file describing what should be compared')
    generate_parser.add_argument('--sequence-length', type=int, default=10, help='seq length')
    generate_parser.add_argument('--no-plot', action='store_true', help='no plot')
    generate_parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test', help='mode')
    generate_parser.add_argument('--n-examples', type=int, default=10, help='number of examples to visualize')
    generate_parser.add_argument('--take', type=int, help='take only a subsect of the data')
    generate_parser.set_defaults(func=generate)

    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('results_dir', type=pathlib.Path)
    evaluate_parser.set_defaults(func=evaluate)

    visualize_parser = subparsers.add_parser('visualize')
    visualize_parser.add_argument('results_dir', type=pathlib.Path)
    visualize_parser.add_argument('--n-examples', type=int, default=10)
    visualize_parser.set_defaults(func=visualize)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
