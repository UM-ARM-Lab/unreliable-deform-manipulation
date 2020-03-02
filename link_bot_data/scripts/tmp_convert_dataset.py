#!/usr/bin/env python
import re
import argparse
import numpy as np

import tensorflow as tf

from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature

tf.compat.v1.enable_eager_execution()
import pathlib
from typing import List

from link_bot_data.base_dataset import BaseDataset


class TmpDataset(BaseDataset):
    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super().__init__(dataset_dirs)
        self.action_like_names_and_shapes = ['%d/action']

        self.state_like_names_and_shapes = [
            '%d/state/local_env',
            '%d/state/local_env_origin',
            '%d/state/link_bot',
            '%d/res',
            '%d/time_idx',
            '%d/traj_idx',
        ]
        self.trajectory_constant_names_and_shapes = [
            'full_env/origin',
            'full_env/extent',
            'full_env/env',
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', nargs='+', type=pathlib.Path)
    parser.add_argument('out_dir', type=pathlib.Path)

    n_examples_per_record = 128
    compression_type = "ZLIB"

    args = parser.parse_args()

    for mode in ['test', 'val', 'train']:
        dataset = TmpDataset(args.dataset_dir)
        tf_dataset = dataset.get_datasets(mode=mode,  n_parallel_calls=1, do_not_process=True)

        full_output_directory = args.out_dir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        current_record_idx = 0
        examples = np.ndarray([n_examples_per_record], dtype=np.object)
        for example_idx, example_dict in enumerate(tf_dataset):

            features = {
                'full_env/res': float_tensor_to_bytes_feature(0.03),
                'local_env/res': float_tensor_to_bytes_feature(0.03),
            }
            for k, v in example_dict.items():
                if 'res' not in k:
                    features[k] = float_tensor_to_bytes_feature(v)

            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            example = example_proto.SerializeToString()
            examples[current_record_idx] = example
            current_record_idx += 1

            if current_record_idx == n_examples_per_record:
                # save to a TF record
                serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

                end_example_idx = example_idx + 1
                start_example_idx = end_example_idx - n_examples_per_record
                record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
                full_filename = full_output_directory / record_filename
                writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=compression_type)
                writer.write(serialized_dataset)
                print("saved {}".format(full_filename))

                current_record_idx = 0


if __name__ == '__main__':
    main()
