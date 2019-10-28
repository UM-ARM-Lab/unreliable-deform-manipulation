#!/usr/bin/env python
import argparse
import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore
from matplotlib.animation import FuncAnimation

from link_bot_planning import model_utils
from video_prediction.datasets import dataset_utils

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


def generate(args):
    dataset_hparams_dict = json.load(open(args.input_dir / 'hparams.json', 'r'))

    ###############
    # Datasets
    ###############
    dataset_hparams_dict['sequence_length'] = args.sequence_length
    dataset, tf_dataset = dataset_utils.get_dataset(args.input_dir,
                                                    'state_space',
                                                    dataset_hparams_dict,
                                                    '',
                                                    shuffle=False,
                                                    mode=args.mode,
                                                    epochs=1,
                                                    seed=0,
                                                    batch_size=1)

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
    results = np.load(args.results_filename)
    evaluate_metrics(results)


def visualize(args):
    results_dir = args.results_filename.parent
    results = np.load(args.results_filename)
    visualize_predictions(results, args.n_examples, results_dir)


def generate_results(outdir, models, tf_dataset, mode):
    results = {
        'true': []
    }
    for x, y in tf_dataset.take(128):
        true_points = y['output_states'].numpy().squeeze().reshape([-1, 3, 2])
        results['true'].append(true_points)

    for model_name, model in models.items():
        results[model_name] = []
        print("generating results for {}".format(model_name))
        for x, y in tf_dataset.take(128):
            states = x['states'].numpy()
            actions = x['actions'].numpy()
            # this is supposed to give us a [batch, n_state] tensor
            first_states = np.expand_dims(states[0, 0], axis=0)
            predicted_points = model.predict(first_states, actions)[0]
            results[model_name].append(predicted_points)

    results_filename = outdir / 'results-{}-{}.npz'.format(mode, int(time.time()))
    print(Fore.CYAN + "Saving results to {}".format(results_filename) + Fore.RESET)
    np.savez(results_filename, **results)
    return results


def visualize_predictions(results, n_examples, outdir=None):
    n_examples = min(len(results['true']), n_examples)
    sequence_length = results['true'][0].shape[0]
    for example_idx in range(n_examples):
        xmin = np.min(results['true'][example_idx][:, :, 0]) - 0.4
        ymin = np.min(results['true'][example_idx][:, :, 1]) - 0.4
        xmax = np.max(results['true'][example_idx][:, :, 0]) + 0.4
        ymax = np.max(results['true'][example_idx][:, :, 1]) + 0.4

        fig, ax = plt.subplots()
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(example_idx)

        time_text_handle = plt.text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5}, bbox=dict(facecolor='black', alpha=0.5))

        # create all the necessary plotting handles
        handles = {}
        for model_name, _ in results.items():
            handles[model_name] = {}
            handles[model_name]['line'] = plt.plot([], [], alpha=0.5, label=model_name)[0]
            handles[model_name]['scatt'] = plt.scatter([], [], s=10)

        def update(t):
            for _model_name, points_trajectories in results.items():
                points = points_trajectories[example_idx][t]
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
    for model_name, points_trajectories in results.items():
        if model_name == 'true':
            continue

        # loop over trajectories
        total_errors = []
        head_errors = []
        tail_errors = []
        mid_errors = []
        for i, predicted_points in enumerate(points_trajectories):
            true_points = results['true'][i]
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


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('input_dir', type=pathlib.Path)
    generate_parser.add_argument('comparison', type=pathlib.Path, help='json file describing what should be compared')
    generate_parser.add_argument('outdir', type=pathlib.Path)
    generate_parser.add_argument('--sequence-length', type=int, default=10)
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
