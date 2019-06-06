#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import time
import tensorflow as tf
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from enum import auto

from attr import dataclass
from matplotlib.lines import Line2D

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
    predicted_violated: bool
    true_violated: bool


class SavableFigureCollection:

    def __init__(self, figures):
        self.figures = figures

    def save(self, filename):
        for i, f in enumerate(self.figures):
            f.save(filename + '_' + i)


class SavableAnimationWrapper:

    def __init__(self, anim, writer):
        self.anim = anim
        self.writer = writer

    def save(self, filename):
        self.anim.save(filename + '.mp4', writer=self.writer)


class SavableFigure:

    def __init__(self, figure):
        self.figure = figure

    def save(self, filename):
        self.figure.savefig(filename)


def make_rope_configuration(head_x, head_y, theta_1, theta_2):
    rope_configuration = np.zeros(6)
    rope_configuration[4] = head_x
    rope_configuration[5] = head_y
    rope_configuration[2] = rope_configuration[4] + np.cos(theta_1)
    rope_configuration[3] = rope_configuration[5] + np.sin(theta_1)
    rope_configuration[0] = rope_configuration[2] + np.cos(theta_2)
    rope_configuration[1] = rope_configuration[3] + np.sin(theta_2)
    return rope_configuration


def get_rope_configurations(args):
    if args.dataset:
        data = np.load(args.dataset)
        states = data['states']
        N = states.shape[2]
        assert N == 6
        rope_configurations = states.reshape(-1, N)
    else:
        m = 1000
        rope_configurations = np.ndarray((m, 6))
        for i in range(m):
            theta_1 = np.random.uniform(-np.pi, np.pi)
            theta_2 = np.random.uniform(-np.pi, np.pi)
            head_x = np.random.uniform(-5, 5)
            head_y = np.random.uniform(-5, 5)
            rope_configurations[i] = make_rope_configuration(head_x, head_y, theta_1, theta_2)
    return rope_configurations


def evaluate_single(sdf, sdf_resolution, sdf_origin, model, threshold, rope_configuration):
    predicted_violated, predicted_point = model.violated(rope_configuration)
    predicted_point = predicted_point.squeeze()
    rope_configuration = rope_configuration.squeeze()
    head_x = rope_configuration[4]
    head_y = rope_configuration[5]
    row_col = link_bot_pycommon.point_to_sdf_idx(head_x, head_y, sdf_resolution, sdf_origin)
    true_violated = sdf[row_col] < threshold

    result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, true_violated)
    return result


def evaluate(sdf, sdf_resolution, sdf_origin, model, threshold, states_flat):
    m = states_flat.shape[0]
    results = np.ndarray([m], dtype=EvaluateResult)
    for i, rope_configuration in enumerate(states_flat):
        rope_configuration = np.atleast_2d(rope_configuration)
        result = evaluate_single(sdf, sdf_resolution, sdf_origin, model, threshold, rope_configuration)
        results[i] = result

    return results


def plot_examples(sdf_image, results, subsample=10, title=''):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(sdf_image, extent=[-5, 5, -5, 5])

    predicted_xs = [result.predicted_point[0] for result in results[::subsample]]
    predicted_ys = [result.predicted_point[1] for result in results[::subsample]]
    head_xs = [result.rope_configuration[4] for result in results[::subsample]]
    head_ys = [result.rope_configuration[5] for result in results[::subsample]]

    plt.scatter(predicted_xs, predicted_ys, s=5, c='r', label='predicted', zorder=2)
    plt.scatter(head_xs, head_ys, s=5, c='b', label='true', zorder=2)

    for result in results[::subsample]:
        predicted_point = result.predicted_point
        rope_configuration = result.rope_configuration
        plt.plot([predicted_point[0], rope_configuration[4]], [predicted_point[1], rope_configuration[5]], c='k',
                 linewidth=1, zorder=1, alpha=0.1)
    plt.legend()
    return SavableFigure(fig)


def plot_single_example(sdf_image, result):
    fig = plt.figure()
    plt.imshow(sdf_image, extent=[-5, 5, -5, 5])
    plt.plot(result.rope_configuration[[0, 2, 4]], result.rope_configuration[[1, 3, 5]], label='rope')

    if result.predicted_violated:
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
    return SavableFigure(fig)


def plot_interpolate(sdf, sdf_resolution, sdf_origin, sdf_image, model, threshold, title=''):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(sdf_image, extent=[-5, 5, -5, 5])

    head_xs = np.linspace(-4.95, 4.95, 25)
    head_ys = np.linspace(-4.95, 4.95, 25)
    theta_1s = np.linspace(-np.pi, np.pi, 2)
    theta_2s = np.linspace(-np.pi, np.pi, 2)
    grid = np.meshgrid(head_xs, head_ys, theta_1s, theta_2s)
    grid = [g.reshape(-1) for g in grid]
    rope_params = np.vstack(grid).T
    rope_configuration_0 = make_rope_configuration(*rope_params[0])

    result_0 = evaluate_single(sdf, sdf_resolution, sdf_origin, model, threshold, rope_configuration_0)
    head_scatter = plt.scatter(result_0.rope_configuration[4], result_0.rope_configuration[5], s=50, c='b', zorder=2)
    prediction_scatter = plt.scatter(result_0.predicted_point[0], result_0.predicted_point[1], s=10, c='r', zorder=2)

    xs_0 = [rope_configuration_0[0], rope_configuration_0[2], rope_configuration_0[4]]
    ys_0 = [rope_configuration_0[1], rope_configuration_0[3], rope_configuration_0[5]]
    line = plt.plot(xs_0, ys_0, color='black', linewidth=2, zorder=1)[0]

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-5.1, 5.1])
    plt.ylim([-5.1, 5.1])

    custom_lines = [
        Line2D([0], [0], color='r', lw=1),
        Line2D([0], [0], color='b', lw=1),
    ]
    plt.legend(custom_lines, ['pred', 'true'])

    def update(t):
        rope_configuration = make_rope_configuration(*rope_params[t])
        result = evaluate_single(sdf, sdf_resolution, sdf_origin, model, threshold, rope_configuration)

        head_scatter.set_offsets(rope_configuration[4:6])

        prediction_scatter.set_offsets(result.predicted_point)

        xs = [rope_configuration[0], rope_configuration[2], rope_configuration[4]]
        ys = [rope_configuration[1], rope_configuration[3], rope_configuration[5]]
        line.set_xdata(xs)
        line.set_ydata(ys)

    fps = 100
    duration_s = 30
    T = rope_params.shape[0]
    interval_ms = int(duration_s * 1000 / T)
    duration_s = T / fps
    print('animation will be {} seconds long'.format(duration_s))
    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800)
    return SavableAnimationWrapper(anim, writer)


def plot(args, sdf, sdf_resolution, sdf_origin, model, threshold, results, true_positives, true_negatives,
         false_positives, false_negatives):
    img = Image.fromarray(np.uint8(np.flipud(sdf.T) > threshold))
    sdf_image = img.resize((200, 200))

    if args.plot_type == PlotType.random_individual:
        random_indexes = np.random.choice(model, size=10, replace=False)
        random_results = results[random_indexes]
        figs = []
        for random_result in random_results:
            fig = plot_single_example(sdf_image, random_result)
            figs.append(fig)
        return SavableFigureCollection(figs)

    elif args.plot_type == PlotType.random_combined:
        random_indeces = np.random.choice(model, size=1000, replace=False)
        random_results = results[random_indeces]
        savable = plot_examples(sdf_image, random_results, subsample=1, title='random samples')
        return savable

    elif args.plot_type == PlotType.true_positives:
        savable = plot_examples(sdf_image, true_positives, subsample=5, title='true positives')
        return savable

    elif args.plot_type == PlotType.true_negatives:
        savable = plot_examples(sdf_image, true_negatives, subsample=5, title='true negatives')
        return savable

    elif args.plot_type == PlotType.false_positives:
        savable = plot_examples(sdf_image, false_positives, subsample=1, title='false positives')
        return savable

    elif args.plot_type == PlotType.false_negatives:
        savable = plot_examples(sdf_image, false_negatives, subsample=1, title='false negatives')
        return savable

    elif args.plot_type == PlotType.interpolate:
        savable = plot_interpolate(sdf, sdf_resolution, sdf_origin, sdf_image, model, threshold, title='interpolate')
        return savable


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=ConstraintModelType.from_string, choices=list(ConstraintModelType))
    parser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    parser.add_argument("checkpoint", help="eval the *.ckpt name")
    parser.add_argument("plot_type", type=PlotType.from_string, choices=list(PlotType))
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--save", action='store_true')
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

    # get the rope configurations we're going to evaluate
    rope_configurations = get_rope_configurations(args)
    m = rope_configurations.shape[0]

    # evaluate the rope configurations
    results = evaluate(sdf, sdf_resolution, sdf_origin, model, threshold, rope_configurations)

    true_positives = np.array([result for result in results if result.true_violated and result.predicted_violated])
    n_true_positives = len(true_positives)
    false_positives = np.array([result for result in results if result.true_violated and not result.predicted_violated])
    n_false_positives = len(false_positives)
    true_negatives = np.array([result for result in results if not result.true_violated and not result.predicted_violated])
    n_true_negatives = len(true_negatives)
    false_negatives = np.array([result for result in results if not result.true_violated and result.predicted_violated])
    n_false_negatives = len(false_negatives)

    accuracy = (n_true_positives + n_true_negatives) / m
    precision = n_true_positives / (n_true_positives + n_false_positives)
    recall = n_true_negatives / (n_true_negatives + n_false_negatives)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)

    savable = plot(args, sdf, sdf_resolution, sdf_origin, model, threshold, results, true_positives, true_negatives,
                   false_positives, false_negatives)

    plt.show()
    if args.save:
        savable.save('plot_constraint_{}-{}'.format(args.plot_type.name, int(time())))


if __name__ == '__main__':
    main()
