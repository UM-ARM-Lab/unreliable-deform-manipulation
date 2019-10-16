#!/usr/bin/env python
import argparse
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.video_prediction_dataset_utils import float_feature

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=pathlib.Path)
    parser.add_argument('--n-examples-per-record', type=int, default=1024)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--pre', type=float, default=0.23)
    parser.add_argument('--post', type=float, default=0.23)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    root_output_directory = args.indir.parent / (args.indir.name + "-fixed")
    root_output_directory.mkdir(exist_ok=True)

    # copy the hparams file
    hparams_path = args.indir / 'hparams.json'
    shutil.copy2(hparams_path, root_output_directory)

    for mode in ['train', 'test', 'val']:
        full_output_directory = root_output_directory / mode
        full_output_directory.mkdir(exist_ok=True)

        classifier_dataset = ClassifierDataset(args.indir)
        dataset = classifier_dataset.get_dataset(mode=mode, num_epochs=1, batch_size=0)

        current_record_idx = 0
        examples = np.ndarray([args.n_examples_per_record], dtype=np.object)
        example_idx = 0
        for example_dict in dataset:
            planned_origin = example_dict['planned_local_env/origin'].numpy()
            planned_fixed_origin = np.flip(planned_origin)
            actual_origin = example_dict['actual_local_env/origin'].numpy()
            actual_fixed_origin = np.flip(actual_origin)
            # TODO: figure out a better way to do this
            features = {
                'actual_local_env/env': float_feature(example_dict['actual_local_env/env'].numpy().flatten()),
                'actual_local_env/extent': float_feature(example_dict['actual_local_env/extent'].numpy()),
                'actual_local_env/origin': float_feature(actual_fixed_origin),
                'planned_local_env/env': float_feature(example_dict['planned_local_env/env'].numpy().flatten()),
                'planned_local_env/extent': float_feature(example_dict['planned_local_env/extent'].numpy()),
                'planned_local_env/origin': float_feature(planned_fixed_origin),
                'res': float_feature(example_dict['res'].numpy()),
                'w_m': float_feature(example_dict['w_m'].numpy()),
                'h_m': float_feature(example_dict['h_m'].numpy()),
                'state': float_feature(example_dict['state'].numpy()),
                'next_state': float_feature(example_dict['next_state'].numpy()),
                'action': float_feature(example_dict['action'].numpy()),
                'planned_state': float_feature(example_dict['planned_state'].numpy()),
                'planned_next_state': float_feature(example_dict['planned_next_state'].numpy()),
                'label': float_feature(example_dict['label'].numpy()),
            }
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


if __name__ == '__main__':
    main()
