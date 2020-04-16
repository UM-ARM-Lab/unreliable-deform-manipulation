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
from link_bot_pycommon.args import my_formatter
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def generate(args):
    ###############
    # Datasets
    ###############
    base_folder = pathlib.Path('results') / 'compare_models-{}-{}'.format(args.mode, int(time.time()))
    base_folder.mkdir(parents=True)

    comparison_info = json.load(args.comparison.open("r"))
    models = {}
    for name, model_info in comparison_info.items():
        model_dir = model_info['model_dir']
        model, _ = model_utils.load_generic_model(model_dir)
        models[name] = model

    dataset = DynamicsDataset(args.dataset_dirs)
    tf_dataset = dataset.get_datasets(mode=args.mode, sequence_length=args.sequence_length, take=args.take)

    results = generate_results(base_folder, models, tf_dataset, args.sequence_length)

    evaluate_metrics(results)

    if not args.no_plot:
        visualize_predictions(results, args.n_examples, base_folder)


def evaluate(args):
    results = pickle.load(args.results_filename.open("rb"))
    evaluate_metrics(results)


def visualize(args):
    results_dir = args.results_filename.parent
    results = pickle.load(args.results_filename.open("rb"))
    visualize_predictions(results, args.n_examples, results_dir)


def generate_results(base_folder: pathlib.Path,
                     models: Dict[str, BaseDynamicsFunction],
                     tf_dataset,
                     sequence_length: int):
    results = {
        'true': {
            'points': [],
            'runtimes': [],
            'full_env/env': [],
            'full_env/extent': [],
        }
    }

    metadata = []
    for model_name, model in models.items():
        model_info = {
            'name': model_name,
            'hparams': model.hparams,
        }
        metadata.append(model_info)

    # Collect ground truth
    for x, y in tf_dataset:
        output_states_dict = y
        pred_link_bot_states = output_states_dict['link_bot'].numpy()
        true_points = pred_link_bot_states.reshape([sequence_length, -1, 2])

        full_env_extent = x['full_env/extent'].numpy()
        full_env = x['full_env/env'].numpy()

        results['true']['points'].append(true_points)
        results['true']['runtimes'].append(np.inf)
        results['true']['full_env/env'].append(full_env)
        results['true']['full_env/extent'].append(full_env_extent)

    # run predictions
    for model_name, model in models.items():
        results[model_name] = {
            'points': [],
            'runtimes': [],
            'full_env/env': [],
            'full_env/extent': [],
        }
        print("generating results for {}".format(model_name))
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
            full_env_extents = x['full_env/extent'].numpy()

            t0 = time.time()

            predictions = model.propagate(full_env=full_envs,
                                          full_env_origin=full_env_origin,
                                          res=res,
                                          start_states=start_states,
                                          actions=actions)
            points = np.array([model.scenario.points_for_compare_models(state) for state in predictions])
            runtime = time.time() - t0

            results[model_name]['points'].append(points)
            results[model_name]['full_env/env'].append(full_envs)
            results[model_name]['full_env/extent'].append(full_env_extents)
            results[model_name]['runtimes'].append(runtime)

    metadata_filename = base_folder / 'metadata.json'
    json.dump(metadata, metadata_filename.open('w'), indent=2)
    results_filename = base_folder / 'results.pkl'
    print(Fore.CYAN + "Saving results to {}".format(results_filename) + Fore.RESET)
    pickle.dump(results, results_filename.open("wb"))
    return results


def visualize_predictions(results, n_examples, base_folder=None):
    n_examples = min(len(results['true']['points']), n_examples)
    sequence_length = results['true']['points'][0].shape[0]
    for example_idx in range(n_examples):
        fig, _ = plt.subplots(constrained_layout=True)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.title(example_idx)

        full_env = results['true']['full_env/env'][example_idx]
        extent = results['true']['full_env/extent'][example_idx]
        min_x = extent[0] * 0.9
        max_y = extent[3] * 0.9
        time_text_handle = plt.text(min_x, max_y, 't=0', fontdict={'color': 'white', 'size': 5},
                                    bbox=dict(facecolor='black', alpha=0.5))
        plt.imshow(np.flipud(full_env), extent=extent)

        plt.xlim(extent[0:2])
        plt.ylim(extent[2:4])

        # create all the necessary plotting handles
        handles = {}
        for model_name, _ in results.items():
            handles[model_name] = {}
            handles[model_name]['line'] = plt.plot([], [], alpha=0.5, label=model_name)[0]
            handles[model_name]['scatt'] = plt.scatter([], [], s=50)

        def update(t):
            for _model_name, points_trajectories in results.items():
                points = points_trajectories['points'][example_idx][t]
                xs = points[:, 0]
                ys = points[:, 1]
                handles[_model_name]['line'].set_xdata(xs)
                handles[_model_name]['line'].set_ydata(ys)
                scatt_coords = np.vstack((xs, ys)).T
                handles[_model_name]['scatt'].set_offsets(scatt_coords)
            time_text_handle.set_text("t={}".format(t))

        plt.legend()

        anim = FuncAnimation(fig, update, frames=sequence_length, interval=500)
        anim_path = base_folder / 'anim-{}.gif'.format(example_idx)
        anim.save(anim_path, writer='imagemagick', fps=4)
        plt.show()


def evaluate_metrics(results):
    sequence_length, n_points, _ = results['true']['points'][0].shape
    for model_name, result in results.items():
        if model_name == 'true':
            continue

        runtimes = result['runtimes']

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

        print()
        print("Model: {}".format(model_name))
        for i in range(n_points):
            print("point {} error:  {:8.4f}m {:6.4f}".format(i, np.mean(errors_by_point[i]), np.std(errors_by_point[i])))
        print("total error: {:8.4f}m {:6.4f}".format(np.mean(total_errors), np.std(total_errors)))
        print("final_tail error: {:8.4f}m {:6.4f}".format(np.mean(final_tail_errors), np.std(final_tail_errors)))
        print("runtime: {:8.4f}ms".format(np.mean(runtimes) * 1e3))


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
