#!/usr/bin/env python
import argparse
import json
import logging
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf

from link_bot_data.classifier_dataset import predict_and_nullify
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import data_directory
from link_bot_planning import model_utils
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_pycommon import compute_max_consecutive_zeros
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('fwd_model_dir', type=pathlib.Path, help='forward model', nargs="+")
    parser.add_argument('--mode', choices=['train', 'test', 'val', 'all'], default='train')
    parser.add_argument('--take', type=int)
    parser.add_argument('--sequence-length', type=int, default=10)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--no-plot', action='store_true')

    args = parser.parse_args()

    tf.random.set_seed(1)

    root = None
    if args.save:
        root = data_directory(pathlib.Path('results/funneling_examples/1'))

    labeling_params = json.load(args.labeling_params.open("r"))
    fwd_models, _ = model_utils.load_generic_model(args.fwd_model_dir)

    dataset = DynamicsDataset([args.dataset_dir])

    tf_dataset = dataset.get_datasets(mode=args.mode, take=args.take, sequence_length=args.sequence_length)

    if args.shuffle:
        tf_dataset = tf_dataset.shuffle(2048)

    maxs_consecutive_zeros = []
    n_funneling = 0
    for example_idx, dataset_element in enumerate(tf_dataset.batch(1)):
        inputs, outputs = dataset_element

        predictions, _ = predict_and_nullify(dataset, fwd_models, dataset_element, labeling_params, 1, 0)


        # check if this example shows funneling (i.e a label of 1 after a label of 0)
        threshold = labeling_params['threshold']
        state_key = labeling_params['state_key']
        pred_sequence_for_state_key = predictions[state_key]
        sequence_for_state_key = outputs[state_key]
        model_error = tf.linalg.norm(sequence_for_state_key - pred_sequence_for_state_key, axis=2)
        labels = tf.cast(model_error < threshold, dtype=tf.int64)
        num_ones = tf.reduce_sum(labels)
        index_of_last_1 = tf.reduce_max(tf.where(labels))
        funneling = (index_of_last_1 >= num_ones)

        if funneling:
            n_funneling += 1
            labels_list = labels.numpy().squeeze()
            # accumulated statistics
            max_consecutive_zeros = compute_max_consecutive_zeros(labels_list)
            maxs_consecutive_zeros.append(max_consecutive_zeros)

            # animate the state versus ground truth
            anim = fwd_models.scenario.animate_predictions_from_dataset(example_idx=example_idx,
                                                                        dataset_element=dataset_element,
                                                                        predictions=predictions,
                                                                        labels=labels_list)
            if args.save:
                filename = root / 'example_{}.gif'.format(example_idx)
                print("Saving {}".format(filename))
                anim.save(filename, writer='imagemagick', dpi=300, fps=1)
            if not args.no_plot:
                plt.show()

    print("{}/{} examples are funneling  [mode={}]".format(n_funneling, example_idx, args.mode))

    if not args.no_plot:
        bins = range(args.sequence_length)
        plt.hist(maxs_consecutive_zeros, bins=bins, align='left')
        plt.xlabel("number of consecutive diverged states (out of 10 possible)")
        plt.ylabel("count")
        plt.xticks(bins)
        plt.show()


if __name__ == '__main__':
    main()