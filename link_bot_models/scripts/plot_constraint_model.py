#!/usr/bin/env python
from __future__ import print_function

import argparse
from time import time

import matplotlib.pyplot as plt
from colorama import Fore, Style
import numpy as np
import tensorflow as tf

from link_bot_models import constraint_sdf
from link_bot_models import plotting
from link_bot_models.constraint_sdf import ConstraintSDF
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset


def plot(args, sdf_data, model, threshold, results, true_positives, true_negatives, false_positives, false_negatives):
    n_examples = results.shape[0]

    sdf_data.image = (sdf_data.image < threshold).astype(np.uint8)

    if args.plot_type == plotting.PlotType.none:
        return None

    if args.plot_type == plotting.PlotType.random_individual:
        random_indexes = np.random.choice(n_examples, size=10, replace=False)
        random_results = results[random_indexes]
        figs = []
        for random_result in random_results:
            fig = plotting.plot_single_example(sdf_data, random_result)
            figs.append(fig)
        return plotting.SavableFigureCollection(figs)

    elif args.plot_type == plotting.PlotType.random_combined:
        random_indeces = np.random.choice(n_examples, size=100, replace=False)
        random_results = results[random_indeces]
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, random_results, subsample=1, title='random samples')
        return savable

    elif args.plot_type == plotting.PlotType.true_positives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, true_positives, subsample=5, title='true positives')
        return savable

    elif args.plot_type == plotting.PlotType.true_negatives:
        savable = plotting.plot_examples(sdf_data.image, sdf_data.extent, true_negatives, subsample=5, title='true negatives')
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
        savable = plotting.plot_contours(sdf_data, model, threshold)
        return savable

    elif args.plot_type == plotting.PlotType.animate_contours:
        savable = plotting.animate_contours(sdf_data, model, threshold)
        return savable


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help='use this dataset instead of random rope configurations')
    parser.add_argument("checkpoint", help="eval the *.ckpt name")
    parser.add_argument("threshold", type=float)
    # parser.add_argument("label_type", type=LabelType.from_string, choices=list(LabelType))
    parser.add_argument("plot_type", type=plotting.PlotType.from_string, choices=list(plotting.PlotType))
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_init", action='store_true')

    args = parser.parse_args()

    label_types = [LabelType.SDF]
    label_mask = LabelType.mask(label_types)

    # get the rope configurations we're going to evaluate
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)

    args_dict = vars(args)
    model = ConstraintSDF(args_dict, dataset.sdf_shape, args.N)
    model.setup()

    for env_idx, environment in enumerate(dataset.environments):
        print(Style.BRIGHT + Fore.GREEN + "Environment {}".format(env_idx) + Fore.RESET + Style.NORMAL)

        sdf_data = environment.sdf_data

        results = constraint_sdf.test_predictions(model, environment)
        m = results.shape[0]

        true_positives = [result for result in results if label_mask @ result.true_violated and result.predicted_violated]
        true_positives = np.array(true_positives)
        n_true_positives = len(true_positives)
        false_positives = [result for result in results if not label_mask @ result.true_violated and result.predicted_violated]
        false_positives = np.array(false_positives)
        n_false_positives = len(false_positives)
        true_negatives = [result for result in results if not label_mask @ result.true_violated and not result.predicted_violated]
        true_negatives = np.array(true_negatives)
        n_true_negatives = len(true_negatives)
        false_negatives = [result for result in results if label_mask @ result.true_violated and not result.predicted_violated]
        false_negatives = np.array(false_negatives)
        n_false_negatives = len(false_negatives)

        accuracy = (n_true_positives + n_true_negatives) / m * 100
        precision = n_true_positives / (n_true_positives + n_false_positives) * 100
        recall = n_true_positives / (n_true_positives + n_false_negatives) * 100
        print("precision: {:4.1f}%".format(precision))
        print("recall: {:4.1f}%".format(recall))
        print(Style.BRIGHT + "accuracy: {:4.1f}%".format(accuracy) + Style.NORMAL)

        savable = plot(args, sdf_data, model, args.threshold, results, true_positives, true_negatives,
                       false_positives, false_negatives)

        plt.show()
        if args.save and savable:
            savable.save('plot_constraint_{}-{}'.format(args.plot_type.name, int(time())))


if __name__ == '__main__':
    main()