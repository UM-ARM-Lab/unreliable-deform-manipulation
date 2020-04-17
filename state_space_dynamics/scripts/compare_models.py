#!/usr/bin/env python
import argparse
import json
import pathlib
import pickle
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore
from matplotlib.animation import FuncAnimation

from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_planning import model_utils
from link_bot_planning.get_scenario import get_scenario
from link_bot_pycommon.args import my_formatter
from moonshine.numpy_utils import dict_of_sequences_to_sequence_of_dicts, dict_of_tensors_to_dict_of_numpy_arrays
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.02)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def generate(args):
    base_folder = pathlib.Path('results') / 'compare_models-{}-{}'.format(args.mode, int(time.time()))
    base_folder.mkdir(parents=True)

    tf_dataset, models = load_dataset_and_models(args)

    results, metadata = generate_results(base_folder, models, tf_dataset)

    evaluate_metrics(results, metadata)

    if not args.no_plot:
        visualize_predictions(results, args.n_examples, base_folder)


def evaluate(args):
    results = pickle.load(args.results_filename.open("rb"))
    evaluate_metrics(results)


def visualize(args):
    results_dir = args.results_filename.parent
    results = pickle.load(args.results_filename.open("rb"))
    visualize_predictions(results, args.n_examples, results_dir)


def load_dataset_and_models(args):
    comparison_info = json.load(args.comparison.open("r"))
    models = {}
    for name, model_info in comparison_info.items():
        model_dir = model_info['model_dir']
        model, _ = model_utils.load_generic_model(model_dir)
        models[name] = model

    dataset = DynamicsDataset(args.dataset_dirs)
    tf_dataset = dataset.get_datasets(mode=args.mode, sequence_length=args.sequence_length, take=args.take)
    return tf_dataset, models


def generate_results(base_folder: pathlib.Path,
                     models: Dict[str, BaseDynamicsFunction],
                     tf_dataset):
    results = {}

    # Collect ground truth
    print("generating results for ground truth")
    results['true'] = generate_ground_truth(tf_dataset)

    # run predictions
    for model_name, model in models.items():
        print("generating results for {}".format(model_name))
        results[model_name] = generate_predictions(model, tf_dataset)

    # Save the results
    results_filename = base_folder / 'results.pkl'
    print(Fore.CYAN + "Saving results to {}".format(results_filename) + Fore.RESET)
    pickle.dump(results, results_filename.open("wb"))

    # Save some metadata
    metadata_filename = base_folder / 'metadata.json'
    metadata = []
    for model_name, model in models.items():
        model_info = {
            'name': model_name,
            'hparams': model.hparams,
        }
        metadata.append(model_info)

    json.dump(metadata, metadata_filename.open('w'), indent=2)

    return results, metadata


def generate_ground_truth(tf_dataset):
    ground_truth = []
    for x, y in tf_dataset:
        y = dict_of_tensors_to_dict_of_numpy_arrays(y)
        y = dict_of_sequences_to_sequence_of_dicts(y)
        ground_truth.append(y)
    return ground_truth


def generate_predictions(model, tf_dataset):
    results = []
    for x, y in tf_dataset:
        actions = x['action'].numpy()

        start_states = {}
        if 'states_keys' in model.hparams:
            states_keys = model.hparams['states_keys']
        else:
            states_keys = [model.hparams['state_key']]

        for state_key in states_keys:
            first_state = x[state_key][0].numpy()
            start_states[state_key] = first_state
        res = x['full_env/res'].numpy()
        if len(res.shape) == 3:
            res = np.squeeze(res, axis=2)
        full_env_origin = x['full_env/origin'].numpy()
        full_envs = x['full_env/env'].numpy()

        predictions = model.propagate(full_env=full_envs,
                                      full_env_origin=full_env_origin,
                                      res=res,
                                      start_states=start_states,
                                      actions=actions)
        results.append(predictions)
    return results


def visualize_predictions(results, n_examples, base_folder=None):
    # results is a dictionary, where each key is the name of the model (or ground truth) and the value is a list of dictionaries
    # each element in the list is an example/trajectory, and each dictionary contains the state predicted by the model

    for example_idx in range(n_examples):
        plt.show()


def evaluate_metrics(results, metadata):
    # sequence_length, n_points, _ = results['true']['points'][0].shape
    for model_name in results.keys():
        if model_name == 'true':
            continue

        # loop over trajectories
        total_errors = []
        final_tail_errors = []
        errors_by_point = [[] for _ in range(n_points)]
        for i, predicted_points in enumerate(result['points']):
            true_points = results['true']['points'][i]
            error = np.linalg.norm(predicted_points - true_points, axis=2)
            final_error = error[-1]
            total_error = np.sum(error, axis=1)
            final_tail_error = final_error[0]
            total_errors.append(total_error)
            final_tail_errors.append(final_tail_error)
            for j in range(n_points):
                error_j = error[:, j]
                errors_by_point[j].extend(error_j)

            # The first time step is copied from ground truth, so it should always have zero error
            assert np.allclose(error[0], 0, atol=1e-5)

        scenario = get_scenario(metadata[model_name])
        metrics_function = scenario.dynamics_metrics_function()
        print()
        print("Model: {}".format(model_name))
        for i in range(n_points):
            print("point {} error:  {:8.4f}m {:6.4f}".format(i, np.mean(errors_by_point[i]), np.std(errors_by_point[i])))
        print("total error: {:8.4f}m {:6.4f}".format(np.mean(total_errors), np.std(total_errors)))
        print("final_tail error: {:8.4f}m {:6.4f}".format(np.mean(final_tail_errors), np.std(final_tail_errors)))


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
    evaluate_parser.add_argument('results_filename', type=pathlib.Path)
    evaluate_parser.set_defaults(func=evaluate)

    visualize_parser = subparsers.add_parser('visualize')
    visualize_parser.add_argument('results_filename', type=pathlib.Path)
    visualize_parser.add_argument('--n-examples', type=int, default=10)
    visualize_parser.set_defaults(func=visualize)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
