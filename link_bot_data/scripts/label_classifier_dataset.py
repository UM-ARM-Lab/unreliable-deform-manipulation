#!/usr/bin/env python
import argparse
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import float_feature

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=pathlib.Path)
    parser.add_argument('--n-examples-per-record', type=int, default=1024)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--pre', type=float, default=0.2)
    parser.add_argument('--post', type=float, default=0.2)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument("--skip", action='store_true', help="do not write labels, only analyze")

    args = parser.parse_args()

    pre_dists = []
    post_dists = []
    positive_labels = 0
    negative_labels = 0

    root_output_directory = args.indir.parent / (args.indir.name + "-labeled")
    root_output_directory.mkdir(exist_ok=True)

    # copy the hparams file
    hparams_path = args.indir / 'hparams.json'
    shutil.copy2(hparams_path, root_output_directory)

    for mode in ['train', 'test', 'val']:
        full_output_directory = root_output_directory / mode
        full_output_directory.mkdir(exist_ok=True)

        classifier_dataset = ClassifierDataset(args.indir)
        dataset = classifier_dataset.get_dataset(mode=mode, batch_size=0, shuffle=False)

        current_record_idx = 0
        examples = np.ndarray([args.n_examples_per_record], dtype=np.object)
        example_idx = 0
        for example_dict in dataset:
            print(example_dict.keys())
            state = example_dict['state'].numpy()
            next_state = example_dict['next_state'].numpy()
            planned_state = example_dict['planned_state'].numpy()
            planned_next_state = example_dict['planned_next_state'].numpy()

            # Compute the label for whether our model should be trusted
            pre_transition_distance = np.linalg.norm(state - planned_state)
            post_transition_distance = np.linalg.norm(next_state - planned_next_state)

            pre_dists.append(pre_transition_distance)
            post_dists.append(post_transition_distance)

            pre_close = pre_transition_distance < args.pre
            post_close = post_transition_distance < args.post

            ###########################################################
            # if pre_close and post_close:
            #     label = 1
            #     positive_labels += 1
            # else:
            #     label = 0
            #     negative_labels += 1

            ###########################################################
            if not pre_close:
                # don't add the example to the labeled training dataset
                continue
            if post_close:
                label = 1
                positive_labels += 1
            else:
                label = 0
                negative_labels += 1
            ###########################################################

            # TODO: figure out a better way to do this
            features = ClassifierDataset.copy_features(example_dict)
            features['label'] = float_feature(np.array([label]))
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            example = example_proto.SerializeToString()
            examples[current_record_idx] = example
            current_record_idx += 1
            example_idx += 1

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

                current_record_idx = 0

    print("Positive labels: {}".format(positive_labels))
    print("Negative labels: {}".format(negative_labels))
    percent_positive = positive_labels / (positive_labels + negative_labels) * 100
    print("Class balance : {:3.2f}% positive".format(percent_positive))

    if not args.no_plot:
        counts, _, _ = plt.hist(pre_dists, label='pre', alpha=0.5, bins=100)
        max_count = np.max(counts)
        plt.hist(post_dists, label='post', alpha=0.5, bins=100)
        plt.plot([args.pre, args.pre], [0, max_count], label='pre threshold')
        plt.plot([args.post, args.post], [0, max_count], label='post threshold', linestyle='--')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
