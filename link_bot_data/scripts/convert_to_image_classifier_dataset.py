#!/usr/bin/env python
import argparse
import json
import pathlib
import shutil
import time

import numpy as np
import tensorflow as tf

from link_bot_classifiers.raster_classifier import RasterClassifier
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import float_feature, balance_by_augmentation
from link_bot_pycommon.link_bot_pycommon import add_bool_arg

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


def main():
    """
    Loads an imbalanced dataset as a classifier dataset, balances it, then saves those to a new format of dataset
    specifically for the classifier
    NOTE:
    A link_bot_state_space_dataset is compatible with a classifier_dataset. However, those datasets are imbalanced and inefficient
    for training the classifier. That's what the new_classifier_dataset is for, and this script converts a classifier_dataset
    to a new_classifier_dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    add_bool_arg(parser, 'balance', required=True, help="upsample negative examples to balance the dataset")
    parser.add_argument('--n-examples-per-record', type=int, default=128)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    now = int(time.time())
    root_output_directory = args.dataset_dir.parent / (args.dataset_dir.name + "-image-classifier-{}".format(now))
    root_output_directory.mkdir(exist_ok=False)

    # copy the hparams file
    hparams_path = args.dataset_dir / 'hparams.json'
    dataset_hparams = json.load(hparams_path.open('r'))
    dataset_hparams['balanced'] = args.balance
    dataset_hparams['type'] = 'image'
    out_hparams_path = root_output_directory / 'hparams.json'
    json.dump(dataset_hparams, out_hparams_path.open('w'), indent=1)

    model_hparams = {
        'classifier_dataset_hparams': {
            'fwd_model_hparams': {
                'dynamics_dataset_hparams': dataset_hparams
            }
        },
        'conv_filters': [],
        'batch_norm': False,
        'fc_layer_sizes': [],
        'dropout_rate': 0,
        'kernel_reg': 0,
        'bias_reg': 0
    }
    batch_size = 1
    net = RasterClassifier(model_hparams, batch_size=batch_size)

    for mode in ['train', 'test', 'val']:
        full_output_directory = root_output_directory / mode
        full_output_directory.mkdir(exist_ok=True)

        # this class maps the sequences down to transitions
        classifier_dataset = ClassifierDataset([args.dataset_dir])
        dataset = classifier_dataset.get_datasets(mode=mode,
                                                  batch_size=batch_size,
                                                  balance_key=None,
                                                  shuffle=False,
                                                  seed=0,
                                                  sequence_length=None)

        def make_image(old_example_dict):
            _, _, _, image = net.make_image(old_example_dict)
            return {
                'image': image,
                'label': old_example_dict['label']
            }

        # convert to images
        image_dataset = dataset.map(make_image)

        if args.balance:
            image_dataset = balance_by_augmentation(image_dataset, 'label')

        current_record_idx = 0
        examples = np.ndarray([args.n_examples_per_record], dtype=np.object)
        for example_idx, new_example_dict in enumerate(image_dataset):
            image = new_example_dict['image']
            numpy_image = image.numpy().flatten()
            features = {
                'label': float_feature(new_example_dict['label'].numpy().flatten()),
                'image': float_feature(numpy_image),
            }

            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            example = example_proto.SerializeToString()
            examples[current_record_idx] = example
            current_record_idx += 1

            if current_record_idx == args.n_examples_per_record:
                # save to a TF record
                serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

                end_example_idx = example_idx + 1
                start_example_idx = end_example_idx - args.n_examples_per_record
                record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
                full_filename = full_output_directory / record_filename
                writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=args.compression_type)
                writer.write(serialized_dataset)
                print("saved {}".format(full_filename))

                current_record_idx = 0


if __name__ == '__main__':
    main()
