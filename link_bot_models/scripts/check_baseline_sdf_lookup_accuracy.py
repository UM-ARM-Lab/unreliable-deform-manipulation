#!/usr/bin/env python

import argparse

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation
from colorama import Fore

from link_bot_data.visualization import plot_rope_configuration
from link_bot_pycommon.link_bot_sdf_utils import point_to_sdf_idx
from video_prediction.datasets import dataset_utils


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=1000)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('indir')
    parser.add_argument('dataset_hparams_dict')

    args = parser.parse_args()

    np.random.seed(0)
    tf.random.set_random_seed(0)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.1))
    gpf.reset_default_session(config=config)
    sess = gpf.get_default_session()

    train_dataset, train_inputs, steps_per_epoch = dataset_utils.get_inputs(args.indir,
                                                                            'link_bot_video',
                                                                            args.dataset_hparams_dict,
                                                                            'sequence_length=100',
                                                                            mode='train',
                                                                            epochs=1,
                                                                            seed=0,
                                                                            batch_size=1,
                                                                            shuffle=False)

    incorrect = 0
    correct = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    while True:
        try:
            data = sess.run(train_inputs)
        except tf.errors.OutOfRangeError:
            print(Fore.RED + "Dataset does not contain {} examples.".format(args.n_training_examples) + Fore.RESET)
            return

        rope_configurations = data['rope_configurations'].squeeze()
        sdfs = np.transpose(data['sdf'].squeeze(), [0, 2, 1])
        resolution = data['sdf_resolution']
        origin = data['sdf_origin']
        constraints = data['constraints']

        for true_violated, upside_down_sdf, rope_config in zip(constraints, sdfs, rope_configurations):
            sdf = np.flipud(upside_down_sdf)
            row, col = point_to_sdf_idx(rope_configurations[4], rope_config[5], resolution=resolution, origin=origin)
            signed_distance = sdf[row, col]
            predicted_violated = signed_distance < 0.02

            if predicted_violated:
                if true_violated:
                    correct += 1
                    tp += 1
                else:
                    incorrect += 1
                    fp += 1
            else:
                if true_violated:
                    incorrect += 1
                    fn += 1
                else:
                    correct += 1
                    tn += 1

    accuracy = correct / (correct + incorrect)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("accuracy: {:5.3f}".format(accuracy))
    print("precision: {:5.3f}".format(precision))
    print("recall: {:5.3f}".format(recall))


if __name__ == '__main__':
    main()
