#!/usr/bin/env python
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from enum import auto

from attr import dataclass

from link_bot_models.constraint_model import ConstraintModel, ConstraintModelType
from link_bot_pycommon import link_bot_pycommon


class PlotType(link_bot_pycommon.ArgsEnum):
    random_individual = auto()
    random_combined = auto()
    true_positives = auto()
    true_negatives = auto()
    false_positives = auto()
    false_negatives = auto()
    interpolate = auto()


@dataclass
class EvaluateResult:
    rope_configuration: np.ndarray
    predicted_point: np.ndarray
    pred_violated: bool
    true_violated: bool


def evaluate(sdf, sdf_resolution, sdf_origin, model, threshold, states_flat):
    m = states_flat.shape[0]
    results = np.ndarray([m], dtype=EvaluateResult)
    for i, rope_configuration in enumerate(states_flat):
        rope_configuration = np.atleast_2d(rope_configuration)
        pred_violated, predicted_point = model.violated(rope_configuration)
        predicted_point = predicted_point.squeeze()
        rope_configuration = rope_configuration.squeeze()
        head_x = rope_configuration[4]
        head_y = rope_configuration[5]
        row_col = link_bot_pycommon.point_to_sdf_idx(head_x, head_y, sdf_resolution, sdf_origin)
        true_violated = sdf[row_col] < threshold

        results[i] = EvaluateResult(rope_configuration, predicted_point, pred_violated, true_violated)

    return results


def plot_examples(sdf_image, results, subsample=10, title=''):
    plt.figure()
    plt.title(title)
    plt.imshow(sdf_image, extent=[-5, 5, -5, 5])

    predicted_xs = [result.predicted_point[0] for result in results[::subsample]]
    predicted_ys = [result.predicted_point[1] for result in results[::subsample]]
    head_xs = [result.rope_configuration[4] for result in results[::subsample]]
    head_ys = [result.rope_configuration[5] for result in results[::subsample]]

    plt.scatter(predicted_xs, predicted_ys, s=5, c='r', label='pred', zorder=2)
    plt.scatter(head_xs, head_ys, s=5, c='b', label='true', zorder=2)

    for result in results[::subsample]:
        predicted_point = result.predicted_point
        rope_configuration = result.rope_configuration
        plt.plot([predicted_point[0], rope_configuration[4]], [predicted_point[1], rope_configuration[5]], c='k',
                 linewidth=1, zorder=1, alpha=0.1)
    plt.legend()


def plot_single_example(sdf_image, result):
    plt.figure()
    plt.imshow(sdf_image, extent=[-5, 5, -5, 5])
    plt.plot(result.rope_configuration[[0, 2, 4]], result.rope_configuration[[1, 3, 5]], label='rope')

    if result.pred_violated:
        pred_color = 'r'
    else:
        pred_color = 'g'

    if result.true_violated:
        true_color = 'r'
    else:
        true_color = 'g'

    plt.scatter(result.predicted_point[0], result.predicted_point[1], s=5, c=pred_color, label='pred')
    plt.scatter(result.rope_configuration[4], result.rope_configuration[5], s=5, c=true_color, label='true')
    plt.legend()


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=ConstraintModelType.from_string, choices=list(ConstraintModelType))
    parser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    parser.add_argument("checkpoint", help="eval the *.ckpt name")
    parser.add_argument("plot_type", type=PlotType.from_string, choices=list(PlotType))
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", help='use this dataset instead of random rope configurations')

    args = parser.parse_args()

    sdf, sdf_gradient, sdf_resolution, sdf_origin = link_bot_pycommon.load_sdf(args.sdf)
    args_dict = vars(args)
    model = ConstraintModel(args_dict, sdf, sdf_gradient, sdf_resolution, sdf_origin, args.N)
    model.setup()

    threshold = 0.2

    img = Image.fromarray(np.uint8(np.flipud(sdf.T) > threshold))
    sdf_image = img.resize((200, 200))

    # get the rope configurations we're going to evaluate
    if args.dataset:
        data = np.load(args.dataset)
        states = data['states']
        N = states.shape[2]
        assert N == 6
        states_flat = states.reshape(-1, N)
        m = states_flat.shape[0]
    else:
        m = 1000
        states_flat = np.ndarray((m, 6))
        for i in range(m):
            rope_configuration = np.zeros(6)
            rope_configuration[4] = np.random.uniform(-5, 5)
            rope_configuration[5] = np.random.uniform(-5, 5)
            theta1 = np.random.uniform(-np.pi, np.pi)
            theta2 = np.random.uniform(-np.pi, np.pi)
            rope_configuration[2] = rope_configuration[4] + np.cos(theta1)
            rope_configuration[3] = rope_configuration[5] + np.sin(theta1)
            rope_configuration[0] = rope_configuration[2] + np.cos(theta2)
            rope_configuration[1] = rope_configuration[3] + np.sin(theta2)
            states_flat[i] = rope_configuration

    # evaluate the rope configurations
    results = evaluate(sdf, sdf_resolution, sdf_origin, model, threshold, states_flat)

    true_positives = np.array([result for result in results if result.true_violated and result.pred_violated])
    n_true_positives = len(true_positives)
    false_positives = np.array([result for result in results if result.true_violated and not result.pred_violated])
    n_false_positives = len(false_positives)
    true_negatives = np.array([result for result in results if not result.true_violated and not result.pred_violated])
    n_true_negatives = len(true_negatives)
    false_negatives = np.array([result for result in results if not result.true_violated and result.pred_violated])
    n_false_negatives = len(false_negatives)

    if args.plot_type == PlotType.random_individual:
        random_indeces = np.random.choice(m, size=10, replace=False)
        random_results = results[random_indeces]
        for random_result in random_results:
            plot_single_example(sdf_image, random_result)

    elif args.plot_type == PlotType.random_combined:
        random_indeces = np.random.choice(m, size=1000, replace=False)
        random_results = results[random_indeces]
        plot_examples(sdf_image, random_results, subsample=1, title='random samples')

    elif args.plot_type == PlotType.true_positives:
        plot_examples(sdf_image, true_positives, subsample=5, title='true positives')

    elif args.plot_type == PlotType.true_negatives:
        plot_examples(sdf_image, true_negatives, subsample=5, title='true negatives')

    elif args.plot_type == PlotType.false_positives:
        plot_examples(sdf_image, false_positives, subsample=1, title='false positives')

    elif args.plot_type == PlotType.false_negatives:
        plot_examples(sdf_image, false_negatives, subsample=1, title='false negatives')

    accuracy = (n_true_positives + n_true_negatives) / m
    precision = n_true_positives / (n_true_positives + n_false_positives)
    recall = n_true_negatives / (n_true_negatives + n_false_negatives)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)

    plt.show()


if __name__ == '__main__':
    main()
