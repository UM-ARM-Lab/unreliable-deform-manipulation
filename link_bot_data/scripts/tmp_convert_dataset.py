#!/usr/bin/env python
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
        self.action_feature_names = ["%d/action"]

        self.state_feature_names = [
            "%d/res",
            "%d/actual_local_env/env",
            "%d/actual_local_env/extent",
            "%d/actual_local_env/origin",
            "%d/planned_local_env/env",
            "%d/planned_local_env/extent",
            "%d/planned_local_env/origin",
            "%d/planned_state",
            "%d/res",
            "%d/state",
            "%d/traj_idx",
            "%d/time_idx ",
        ]

        self.constant_feature_names = [
            'full_env/env',
            'full_env/extent',
            'full_env/origin',
            'local_env_rows',
            'local_env_cols',
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
        tf_dataset = dataset.get_datasets(mode=mode, n_parallel_calls=1, do_not_process=True)

        full_output_directory = args.out_dir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        current_record_idx = 0
        examples = np.ndarray([n_examples_per_record], dtype=np.object)
        for example_idx, example_dict in enumerate(tf_dataset):

            features = {
                'full_env/res': float_tensor_to_bytes_feature(example_dict['0/res']),
                'full_env/env': float_tensor_to_bytes_feature(example_dict['full_env/env']),
                'full_env/extent': float_tensor_to_bytes_feature(example_dict['full_env/extent']),
                'full_env/origin': float_tensor_to_bytes_feature(example_dict['full_env/origin']),
            }
            for k, v in example_dict.items():
                if 'res' in k:
                    continue
                elif 'local_env' in k:
                    continue
                elif 'planned_state' in k:
                    new_k = k + '/link_bot'
                elif 'state' in k:
                    num, _ = k.split("/")
                    new_k = num + "/link_bot"
                elif 'time' in k:
                    num, _ = k.split("/")
                    new_k = num + '/time_idx'
                else:
                    new_k = k
                features[new_k] = float_tensor_to_bytes_feature(v)

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
