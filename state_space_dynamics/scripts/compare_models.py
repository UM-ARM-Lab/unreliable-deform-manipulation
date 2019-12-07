#!/usr/bin/env python
import pickle
import argparse
import json
import pathlib
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore
from matplotlib.animation import FuncAnimation

from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_planning import model_utils
from link_bot_pycommon import link_bot_sdf_utils
from state_space_dynamics.base_forward_model import BaseForwardModel

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def generate(args):
    ###############
    # Datasets
    ###############
    dataset = LinkBotStateSpaceDataset(args.dataset_dirs)
    tf_dataset = dataset.get_datasets(mode=args.mode,
                                      shuffle=False,
                                      seed=0,
                                      batch_size=1,
                                      sequence_length=args.sequence_length)

    comparison_info = json.load(args.comparison.open("r"))
    models = {}
    for name, model_info in comparison_info.items():
        model_dir = pathlib.Path(model_info['model_dir'])
        model_type = model_info['model_type']
        model, _ = model_utils.load_generic_model(model_dir, model_type)
        models[name] = model

    results = generate_results(args.outdir, models, tf_dataset, args.mode)

    evaluate_metrics(results)

    if not args.no_plot:
        visualize_predictions(results, args.n_examples, args.outdir)


def evaluate(args):
    results = pickle.load(args.results_filename.open("rb"))
    evaluate_metrics(results)


def visualize(args):
    results_dir = args.results_filename.parent
    results = pickle.load(args.results_filename.open("rb"))
    visualize_predictions(results, args.n_examples, results_dir)


def generate_results(outdir: pathlib.Path,
                     models: Dict[str, BaseForwardModel],
                     tf_dataset,
                     mode: str):
    results = {
        'true': {
            'points': [],
            'runtimes': [],
            'local_env/env': [],
            'local_env/extent': [],
        }
    }
    for x, y in tf_dataset:
        true_points = y['output_states'].numpy().squeeze().reshape([-1, 3, 2])
        extent = x['actual_local_env_s/extent'][:, 0].numpy()
        local_env = tf.expand_dims(x['actual_local_env_s/env'][:, 0], axis=1).numpy()

        results['true']['points'].append(true_points)
        results['true']['runtimes'].append(np.inf)
        results['true']['local_env/env'].append(local_env)
        results['true']['local_env/extent'].append(extent)

    for model_name, model in models.items():
        results[model_name] = {
            'points': [],
            'runtimes': [],
            'local_env/env': [],
            'local_env/extent': [],
        }
        print("generating results for {}".format(model_name))
        for x, y in tf_dataset:
            states = x['state_s'].numpy()
            actions = x['action_s'].numpy()
            # this is supposed to give us a [batch, n_state] tensor
            first_state = np.expand_dims(states[0, 0], axis=0)
            local_env = tf.expand_dims(x['actual_local_env_s/env'][:, 0], axis=1).numpy()
            res = x['resolution_s'][:, 0].numpy()
            res_2d = np.concatenate([res, res], axis=1)
            origin = x['actual_local_env_s/origin'][:, 0].numpy()
            extent = x['actual_local_env_s/extent'][:, 0].numpy()
            local_env_data = link_bot_sdf_utils.unbatch_occupancy_data(data=local_env, resolution=res_2d, origin=origin)
            t0 = time.time()
            predicted_points = model.predict(local_env_data=local_env_data,
                                             state=first_state,
                                             actions=actions)[0]
            runtime = time.time() - t0
            # TODO: save and visualize the local environment
            results[model_name]['points'].append(predicted_points)
            results[model_name]['local_env/env'].append(local_env)
            results[model_name]['local_env/extent'].append(extent)
            results[model_name]['runtimes'].append(runtime)

    results_filename = outdir / 'results-{}-{}.pkl'.format(mode, int(time.time()))
    print(Fore.CYAN + "Saving results to {}".format(results_filename) + Fore.RESET)
    pickle.dump(results, results_filename.open("wb"))
    return results


def visualize_predictions(results, n_examples, outdir=None):
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
        local_env = results['true']['local_env/env'][example_idx][0, 0]
        extent = results['true']['local_env/extent'][example_idx][0]
        plt.imshow(np.flipud(local_env), extent=extent)

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
        anim_path = outdir / 'anim-{}-{}.gif'.format(int(time.time()), example_idx)
        anim.save(anim_path, writer='imagemagick', fps=4)
        plt.show()


def evaluate_metrics(results):
    for model_name, result in results.items():
        if model_name == 'true':
            continue

        runtimes = result['runtimes']

        # loop over trajectories
        total_errors = []
        head_errors = []
        tail_errors = []
        mid_errors = []
        for i, predicted_points in enumerate(result['points']):
            true_points = results['true']['points'][i]
            error = np.linalg.norm(predicted_points - true_points, axis=2)
            total_error = np.sum(error, axis=1)
            tail_error = error[:, 0]
            mid_error = error[:, 1]
            head_error = error[:, 2]
            total_errors.append(total_error)
            head_errors.append(head_error)
            mid_errors.append(mid_error)
            tail_errors.append(tail_error)

            # The first time step is copied from ground truth, so it should always have zero error
            assert np.all(error[0] == 0)

        print()
        print("Model: {}".format(model_name))
        print("head error:  {:8.4f}m".format(np.mean(head_errors)))
        print("mid error:   {:8.4f}m".format(np.mean(mid_errors)))
        print("tail error:  {:8.4f}m".format(np.mean(tail_errors)))
        print("total error: {:8.4f}m".format(np.mean(total_errors)))
        print("runtime: {:8.4f}ms".format(np.mean(runtimes) * 1e3))


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    generate_parser.add_argument('comparison', type=pathlib.Path, help='json file describing what should be compared')
    generate_parser.add_argument('outdir', type=pathlib.Path)
    generate_parser.add_argument('--sequence-length', type=int, default=25)
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
