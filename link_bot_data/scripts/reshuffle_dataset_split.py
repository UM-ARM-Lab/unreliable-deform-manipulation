#!/usr/bin/env python
import argparse
import pathlib
import re
import shutil

import numpy as np
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import float_feature

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


def count(input_dir, mode):
    pattern = re.compile(r"example_(\d+)_to_(\d+).tfrecords")
    count = 0
    n_exmaples_per_record = None
    full_output_directory = input_dir / mode
    assert full_output_directory.is_dir()
    for filename in full_output_directory.glob("*.tfrecords".format(mode)):
        match = re.findall(pattern, str(filename))
        start_idx = int(match[0][0])
        end_idx = int(match[0][1])
        count += (end_idx - start_idx) + 1
        if n_exmaples_per_record is None:
            n_exmaples_per_record = end_idx - start_idx + 1
    return count, n_exmaples_per_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=pathlib.Path)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    # Figure out how many examples to put in train/test/val
    n_test, n_examples_per_record = count(args.input_dir, 'test')
    n_train, _ = count(args.input_dir, 'train')
    n_val, _ = count(args.input_dir, 'val')

    # Load the entire dataset, all modes
    classifier_dataset = ClassifierDataset(args.input_dir)
    train_dataset = classifier_dataset.get_dataset(mode='train', num_epochs=1, batch_size=0)
    val_dataset = classifier_dataset.get_dataset(mode='val', num_epochs=1, batch_size=0)
    test_dataset = classifier_dataset.get_dataset(mode='test', num_epochs=1, batch_size=0)
    tmp_dataset = train_dataset.concatenate(val_dataset)
    full_dataset = tmp_dataset.concatenate((test_dataset))
    full_dataset = full_dataset.shuffle(buffer_size=1024)

    new_train_dataset = full_dataset.take(n_train)
    minus_train_dataset = full_dataset.skip(n_train)
    new_val_dataset = minus_train_dataset.take(n_val)
    new_test_dataset = minus_train_dataset.skip(n_val)

    root_output_directory = args.input_dir.parent / ('r-' + args.input_dir.name)
    root_output_directory.mkdir(exist_ok=True)

    # copy the hparams file
    hparams_path = args.input_dir / 'hparams.json'
    shutil.copy2(hparams_path, root_output_directory)

    for mode, dataset in [('train', new_train_dataset), ('val', new_val_dataset), ('test', new_test_dataset)]:
        full_output_directory = root_output_directory / mode
        full_output_directory.mkdir(exist_ok=True)
        current_record_idx = 0
        examples = np.ndarray([n_examples_per_record], dtype=np.object)
        example_idx = 0
        for example_dict in dataset:
            features = {
                'actual_local_env/env': float_feature(example_dict['actual_local_env/env'].numpy().flatten()),
                'actual_local_env/extent': float_feature(example_dict['actual_local_env/extent'].numpy()),
                'actual_local_env/origin': float_feature(example_dict['actual_local_env/origin'].numpy()),
                'planned_local_env/env': float_feature(example_dict['planned_local_env/env'].numpy().flatten()),
                'planned_local_env/extent': float_feature(example_dict['planned_local_env/extent'].numpy()),
                'planned_local_env/origin': float_feature(example_dict['planned_local_env/origin'].numpy()),
                'res': float_feature(example_dict['res'].numpy()),
                'w_m': float_feature(example_dict['w_m'].numpy()),
                'h_m': float_feature(example_dict['h_m'].numpy()),
                'state': float_feature(example_dict['state'].numpy()),
                'next_state': float_feature(example_dict['next_state'].numpy()),
                'action': float_feature(example_dict['action'].numpy()),
                'planned_state': float_feature(example_dict['planned_state'].numpy()),
                'planned_next_state': float_feature(example_dict['planned_next_state'].numpy()),
                'label': float_feature(example_dict['label'].numpy())
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            example = example_proto.SerializeToString()
            examples[current_record_idx] = example
            current_record_idx += 1
            example_idx += 1

            if current_record_idx == n_examples_per_record:
                # save to a TF record
                serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

                end_example_idx = example_idx
                start_example_idx = end_example_idx - n_examples_per_record
                record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
                full_filename = full_output_directory / record_filename
                writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=args.compression_type)
                writer.write(serialized_dataset)
                print("saved {}".format(full_filename))

                current_record_idx = 0


if __name__ == '__main__':
    main()
