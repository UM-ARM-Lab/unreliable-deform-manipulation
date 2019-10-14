#!/usr/bin/env python
from __future__ import print_function

import argparse
from dataclasses import dataclass
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Style

from link_bot_classifiers import plotting
from link_bot_classifiers.raster_cnn_model import RasterCNNModelRunner
from link_bot_classifiers.sdf_function_model import EvaluateResult
from link_bot_pycommon import link_bot_sdf_utils
from video_prediction.datasets import dataset_utils


@dataclass
class EvaluateResult:
    rope_configuration: np.ndarray
    predicted_violated: np.ndarray
    true_violated: np.ndarray
    sdf: np.ndarray
    extent: np.ndarray
    image: np.ndarray


def plot(args, model, results, true_positives, true_negatives, false_positives, false_negatives):
    n_examples = results.shape[0]

    if args.plot_type == plotting.PlotType.none:
        return None

    if args.plot_type == plotting.PlotType.random_individual:
        s = min(10, n_examples)
        random_indexes = np.random.choice(n_examples, size=s, replace=False)
        random_results = results[random_indexes]
        figs = []
        for random_result in random_results:
            fig = plotting.plot_single_example(random_result)
            figs.append(fig)
        return plotting.SavableFigureCollection(figs)

    elif args.plot_type == plotting.PlotType.random_combined:
        s = min(100, n_examples)
        random_indeces = np.random.choice(n_examples, size=s, replace=False)
        random_results = results[random_indeces]
        savable = plotting.plot_examples(random_results, subsample=1, title='random samples')
        return savable

    elif args.plot_type == plotting.PlotType.true_positives:
        savable = plotting.plot_examples(true_positives, subsample=5, title='true positives')
        return savable

    elif args.plot_type == plotting.PlotType.true_negatives:
        savable = plotting.plot_examples(true_negatives, subsample=5, title='true negatives')
        return savable

    elif args.plot_type == plotting.PlotType.false_positives:
        savable = plotting.plot_examples(false_positives, subsample=1, title='false positives')
        return savable

    elif args.plot_type == plotting.PlotType.false_negatives:
        savable = plotting.plot_examples(false_negatives, subsample=1, title='false negatives')
        return savable

    elif args.plot_type == plotting.PlotType.interpolate:
        savable = plotting.plot_interpolate(model, title='interpolate')
        return savable

    elif args.plot_type == plotting.PlotType.contours:
        savable = plotting.plot_contours(results, model)
        return savable

    elif args.plot_type == plotting.PlotType.animate_contours:
        savable = plotting.animate_contours(results, model)
        return savable


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("plot_type", type=plotting.PlotType.from_string, choices=list(plotting.PlotType))
    parser.add_argument("input_dir", help="directory of tfrecords")
    parser.add_argument("dataset", type=str, help="dataset class name")
    parser.add_argument("dataset_hparams_dict", type=str, help="json file of hyperparameters")
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset-hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)

    dataset, inputs, steps_per_epoch = dataset_utils.get_inputs(args.input_dir,
                                                                args.dataset,
                                                                args.dataset_hparams_dict,
                                                                args.dataset_hparams,
                                                                mode='test',
                                                                epochs=1,
                                                                seed=args.seed,
                                                                batch_size=args.batch_size)
    model = RasterCNNModelRunner.load(args.checkpoint, inputs, steps_per_epoch)
    sess = tf.Session()
    all_inputs = {}
    while True:
        try:
            inputs_batch = sess.run(inputs)
            for key, values in inputs_batch.items():
                if key not in all_inputs:
                    all_inputs[key] = values
                else:
                    all_inputs[key] = np.concatenate((all_inputs[key], values))
        except tf.errors.OutOfRangeError:
            break
    predictions = model.violated()

    results = []
    inputs_and_predictions = zip(all_inputs['rope_configurations'],
                                 all_inputs['constraints'],
                                 all_inputs['images'],
                                 all_inputs['sdf_resolution'],
                                 all_inputs['sdf_origin'],
                                 all_inputs['sdf'],
                                 predictions)
    for rope_configuration, true_violated, image, res, origin, sdf, prediction in inputs_and_predictions:
        extent = link_bot_sdf_utils.bounds(sdf, resolution=res, origin=origin)
        result = EvaluateResult(rope_configuration=rope_configuration,
                                predicted_violated=prediction,
                                true_violated=true_violated,
                                sdf=sdf,
                                extent=extent,
                                image=image)
        results.append(result)
    results = np.array(results)

    true_positives = [result for result in results if result.true_violated and result.predicted_violated]
    true_positives = np.array(true_positives)
    n_true_positives = len(true_positives)
    false_positives = [result for result in results if not result.true_violated and result.predicted_violated]
    false_positives = np.array(false_positives)
    n_false_positives = len(false_positives)
    true_negatives = [result for result in results if not result.true_violated and not result.predicted_violated]
    true_negatives = np.array(true_negatives)
    n_true_negatives = len(true_negatives)
    false_negatives = [result for result in results if result.true_violated and not result.predicted_violated]
    false_negatives = np.array(false_negatives)
    n_false_negatives = len(false_negatives)

    m = len(results)
    accuracy = (n_true_positives + n_true_negatives) / m * 100
    try:
        precision = n_true_positives / (n_true_positives + n_false_positives) * 100
        print("precision: {:4.1f}%".format(precision))
    except ZeroDivisionError:
        pass
    try:
        recall = n_true_positives / (n_true_positives + n_false_negatives) * 100
        print("recall: {:4.1f}%".format(recall))
    except ZeroDivisionError:
        pass
    print(Style.BRIGHT + "accuracy: {:4.1f}%".format(accuracy) + Style.NORMAL)

    savable = plot(args, model, results, true_positives, true_negatives, false_positives, false_negatives)

    plt.show()
    if args.save and savable:
        savable.save('plot_constraint_{}-{}'.format(args.plot_type.name, int(time())))


if __name__ == '__main__':
    main()
