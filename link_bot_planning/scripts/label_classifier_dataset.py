#!/usr/bin/env python
import argparse
import pathlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from link_bot_data import random_environment_data_utils
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.video_prediction_dataset_utils import float_feature

tf.enable_eager_execution()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=pathlib.Path)
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pre', type=float, default=1)
    parser.add_argument('--post', type=float, default=1)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    classifier_dataset = ClassifierDataset(args.indir)
    dataset = classifier_dataset.get_dataset(mode=args.mode,
                                             shuffle=args.shuffle,
                                             num_epochs=1,
                                             seed=args.seed,
                                             compression_type=args.compression_type)

    pre_dists = []
    post_dists = []

    full_output_directory = args.indir.parent / (args.indir.name + "-labeled")
    print(full_output_directory)

    current_record_idx = 0
    examples = np.ndarray()
    for example_idx, example_dict in enumerate(dataset):
        state = example_dict['state'].numpy()
        next_state = example_dict['next_state'].numpy()
        action = example_dict['action'].numpy()
        planned_state = example_dict['planned_state'].numpy()
        planned_next_state = example_dict['planned_next_state'].numpy()

        # TODO: Try filtering out any examples where the start state is very far apart?

        # Compute the label for whether our model should be trusted
        pre_transition_distance = np.linalg.norm(state - planned_state)
        post_transition_distance = np.linalg.norm(next_state - planned_next_state)

        pre_dists.append(pre_transition_distance)
        post_dists.append(post_transition_distance)

        if pre_transition_distance > args.pre or post_transition_distance > args.post:
            label = 0
        else:
            label = 1

        features = dict((k, float_feature(v)) for k, v in example_dict)
        print(features)
        features['label'] = float_feature(np.array([label]))
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        example = example_proto.SerializeToString()

        if current_record_idx == args.n_examples_per_record:
            # save to a TF record
            serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

            end_example_idx = example_idx
            start_example_idx = end_example_idx - args.n_examples_per_record
            record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
            full_filename = full_output_directory / record_filename
            writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=args.compression_type)
            writer.write(serialized_dataset)
            print("saved {}".format(full_filename))

    plt.hist(pre_dists, label='pre')
    plt.hist(post_dists, label='post')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
