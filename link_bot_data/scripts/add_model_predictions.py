#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_data.base_dataset import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT
from link_bot_data.classifier_dataset import add_model_predictions
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature
from link_bot_pycommon.args import my_formatter
from link_bot_planning.model_utils import load_ensemble

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('fwd_model_dir', type=pathlib.Path, help='forward model', nargs="+")
    parser.add_argument('--n-examples-per-record', type=int, default=128, help="examples per file")
    parser.add_argument('--total-take', type=int, help="will be split up between train/test/val")
    parser.add_argument('out_dir', type=pathlib.Path, help='out dir')

    args = parser.parse_args()

    dynamics_hparams = json.load((args.dataset_dir / 'hparams.json').open('r'))
    fwd_models, _ = load_ensemble(args.fwd_model_dir)
    compression_type = "ZLIB"

    dataset = DynamicsDataset([args.dataset_dir])

    args.out_dir.mkdir(parents=False, exist_ok=False)
    new_hparams_filename = args.out_dir / 'hparams.json'
    classifier_dataset_hparams = dynamics_hparams
    if isinstance(args.fwd_model_dir, list):
        fwd_model_dir = [str(d) for d in args.fwd_model_dir]
    else:
        fwd_model_dir = str(args.fwd_model_dir
    classifier_dataset_hparams['fwd_model_dir'] = fwd_model_dir
    classifier_dataset_hparams['fwd_model_hparams'] = fwd_models[0].hparams
    classifier_dataset_hparams['actual_state_keys'] = dataset.state_feature_names
    classifier_dataset_hparams['planned_state_keys'] = fwd_models[0].states_keys
    json.dump(classifier_dataset_hparams, new_hparams_filename.open("w"), indent=1)

    val_split = int(args.total_take * DEFAULT_VAL_SPLIT) if args.total_take is not None else None
    test_split = int(args.total_take * DEFAULT_TEST_SPLIT) if args.total_take is not None else None
    train_split = args.total_take - val_split - test_split if args.total_take is not None else None
    take_split = {
        'test': test_split,
        'val': val_split,
        'train': train_split
    }

    for mode in ['test', 'val', 'train']:
        tf_dataset = dataset.get_datasets(mode=mode, take=take_split[mode])
        new_tf_dataset = add_model_predictions(fwd_models, tf_dataset, dataset)

        full_output_directory = args.out_dir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        current_record_idx = 0
        examples = np.ndarray([args.n_examples_per_record], dtype=np.object)
        for example_idx, example_dict in enumerate(new_tf_dataset):

            features = {}
            for k, v in example_dict.items():
                features[k] = float_tensor_to_bytes_feature(v)

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
                if full_filename.exists():
                    print(Fore.RED + "Error! Output file {} exists. Aborting.".format(full_filename) + Fore.RESET)
                    return
                writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=compression_type)
                writer.write(serialized_dataset)
                print("saved {}".format(full_filename))

                current_record_idx = 0


if __name__ == '__main__':
    main()
