#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import time
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from link_bot_models import constraint_model
from link_bot_models.constraint_model import ConstraintModel, ConstraintModelType
from link_bot_models import plotting
from link_bot_pycommon import link_bot_pycommon


def get_rope_configurations(args):
    if args.dataset:
        data = np.load(args.dataset)
        states = data['states']
        N = states.shape[2]
        assert N == 6
        rope_configurations = states.reshape(-1, N)
    else:
        m = 10000
        rope_configurations = np.ndarray((m, 6))
        for i in range(m):
            theta_1 = np.random.uniform(-np.pi, np.pi)
            theta_2 = np.random.uniform(-np.pi, np.pi)
            head_x = np.random.uniform(-5, 5)
            head_y = np.random.uniform(-5, 5)
            rope_configurations[i] = link_bot_pycommon.make_rope_configuration(head_x, head_y, theta_1, theta_2)
    return rope_configurations


def plot(args, sdf_data, model, threshold, results, true_positives, true_negatives, false_positives, false_negatives):
    n_examples = results.shape[0]

    if args.plot_type == plotting.PlotType.random_individual:
        random_indexes = np.random.choice(n_examples, size=10, replace=False)
        random_results = results[random_indexes]
        figs = []
        for random_result in random_results:
            fig = plotting.plot_single_example(sdf_data.image, random_result)
            figs.append(fig)
        return plotting.SavableFigureCollection(figs)

    elif args.plot_type == plotting.PlotType.random_combined:
        random_indeces = np.random.choice(n_examples, size=1000, replace=False)
        random_results = results[random_indeces]
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, random_results, subsample=1, title='random samples')
        return savable

    elif args.plot_type == plotting.PlotType.true_positives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, true_positives, subsample=1, title='true positives')
        return savable

    elif args.plot_type == plotting.PlotType.true_negatives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, true_negatives, subsample=1, title='true negatives')
        return savable

    elif args.plot_type == plotting.PlotType.false_positives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, false_positives, subsample=1, title='false positives')
        return savable

    elif args.plot_type == plotting.PlotType.false_negatives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, false_negatives, subsample=1, title='false negatives')
        return savable

    elif args.plot_type == plotting.PlotType.interpolate:
        savable = plotting.plot_interpolate(sdf_data, sdf_data.image, model, threshold, title='interpolate')
        return savable

    elif args.plot_type == plotting.PlotType.contours:
        savable = plotting.plot_contours(sdf_data, sdf_data.image, model)
        return savable

    elif args.plot_type == plotting.PlotType.animate_contours:
        savable = plotting.animate_contours(sdf_data, sdf_data.image, model)
        return savable


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=ConstraintModelType.from_string, choices=list(ConstraintModelType))
    parser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    parser.add_argument("checkpoint", help="eval the *.ckpt name")
    parser.add_argument("plot_type", type=plotting.PlotType.from_string, choices=list(plotting.PlotType))
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("--dataset", help='use this dataset instead of random rope configurations')

    args = parser.parse_args()

    sdf_data = link_bot_pycommon.load_sdf_data(args.sdf)
    args_dict = vars(args)
    model = ConstraintModel(args_dict, sdf_data, args.N)
    model.setup()

    threshold = 0.2

    # get the rope configurations we're going to evaluate
    rope_configurations = get_rope_configurations(args)
    m = rope_configurations.shape[0]

    # evaluate the rope configurations
    results = constraint_model.evaluate(sdf_data, model, threshold, rope_configurations)

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

    savable = plot(args, sdf_data, model, threshold, results, true_positives, true_negatives,
                   false_positives, false_negatives)

    plt.show()
    if args.save:
        savable.save('plot_constraint_{}-{}'.format(args.plot_type.name, int(time())))


if __name__ == '__main__':
    main()
