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

from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_planning import model_utils
from link_bot_pycommon.args import my_formatter
from state_space_dynamics.base_forward_model import BaseForwardModel

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def generate(args):
    ###############
    # Datasets
    ###############
    base_folder = args.outdir / 'compare_models-{}-{}'.format(args.mode, int(time.time()))
    base_folder.mkdir()

    comparison_info = json.load(args.comparison.open("r"))
    models = {}
    for name, model_info in comparison_info.items():
        model_dir = pathlib.Path(model_info['model_dir'])
        model_type = model_info['model_type']
        model, _ = model_utils.load_generic_model(model_dir, model_type)
        models[name] = model

    dataset = LinkBotStateSpaceDataset(args.dataset_dirs)
    tf_dataset = dataset.get_datasets(mode=args.mode, sequence_length=args.sequence_length)
    tf_dataset = tf_dataset.batch(1)

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
                     models: Dict[str, BaseForwardModel],
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

        full_env_extent = x['full_env/extent'][0].numpy()
        full_env = x['full_env/env'][0].numpy()

        results['true']['points'].append(true_points)
        results['true']['runtimes'].append(np.inf)
        results['true']['full_env/env'].append(full_env)
        results['true']['full_env/extent'].append(full_env_extent)

    for model_name, model in models.items():
        results[model_name] = {
            'points': [],
            'runtimes': [],
            'full_env/env': [],
            'full_env/extent': [],
        }
        print("generating results for {}".format(model_name))
        for x, y in tf_dataset:
            states = x['state/link_bot'].numpy()
            actions = x['action'].numpy()

            first_state = states[:, 0]
            res = x['res'].numpy()
            if len(res.shape) == 3:
                res = np.squeeze(res, axis=2)
            full_env_origin = x['full_env/origin'].numpy()
            full_envs = x['full_env/env'].numpy()
            full_env_extents = x['full_env/extent'].numpy()

            t0 = time.time()

            # take in a list of state arrays, according to whatever model hparams says
            predicted_points = model.predict(full_env=full_envs,
                                             full_env_origin=full_env_origin,
                                             res=res,
                                             first_states=first_states,
                                             actions=actions)[0]
            runtime = time.time() - t0

            results[model_name]['points'].append(predicted_points)
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
        xmin = np.min(results['true']['points'][example_idx][:, :, 0]) - 1
        ymin = np.min(results['true']['points'][example_idx][:, :, 1]) - 1
        xmax = np.max(results['true']['points'][example_idx][:, :, 0]) + 1
        ymax = np.max(results['true']['points'][example_idx][:, :, 1]) + 1

        fig, _ = plt.subplots()
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(example_idx)

        time_text_handle = plt.text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5}, bbox=dict(facecolor='black', alpha=0.5))
        full_env = results['true']['full_env/env'][example_idx]
        extent = results['true']['full_env/extent'][example_idx]
        plt.imshow(np.flipud(full_env), extent=extent)

        # create all the necessary plotting handles
        handles = {}
        for model_name, _ in results.items():
            handles[model_name] = {}
            handles[model_name]['line'] = plt.plot([], [], alpha=0.5, label=model_name)[0]
            handles[model_name]['scatt'] = plt.scatter([], [], s=10)

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

        anim = FuncAnimation(fig, update, frames=sequence_length, interval=100)
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
        errors_by_point = [[] for _ in range(n_points)]
        for i, predicted_points in enumerate(result['points']):
            true_points = results['true']['points'][i]
            error = np.linalg.norm(predicted_points - true_points, axis=2)
            total_error = np.sum(error, axis=1)
            total_errors.append(total_error)
            for j in range(n_points):
                error_j = error[:, j]
                errors_by_point[j].extend(error_j)

            # The first time step is copied from ground truth, so it should always have zero error
            assert np.all(error[0] == 0)

        print()
        print("Model: {}".format(model_name))
        for i in range(n_points):
            print("point {} error:  {:8.4f}m {:6.4f}".format(i, np.mean(errors_by_point[i]), np.std(errors_by_point[i])))
        print("total error: {:8.4f}m {:6.4f}".format(np.mean(total_errors), np.std(total_errors)))
        print("runtime: {:8.4f}ms".format(np.mean(runtimes) * 1e3))


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)

    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    generate_parser.add_argument('comparison', type=pathlib.Path, help='json file describing what should be compared')
    generate_parser.add_argument('outdir', type=pathlib.Path)
    generate_parser.add_argument('--sequence-length', type=int, default=50)
    generate_parser.add_argument('--no-plot', action='store_true')
    generate_parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test')
    generate_parser.add_argument('--n-examples', type=int, default=10)
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
