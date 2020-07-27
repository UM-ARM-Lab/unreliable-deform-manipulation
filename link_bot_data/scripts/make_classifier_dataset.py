#!/usr/bin/env python
import argparse
import json
import logging
import pathlib
from time import perf_counter

import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_data.base_dataset import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT
from link_bot_data.classifier_dataset_utils import generate_classifier_examples
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.filesystem_utils import mkdir_and_ask
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import index_dict_of_batched_vectors_tf
from state_space_dynamics import model_utils

limit_gpu_mem(6)


def main():
    rospy.init_node("make_classifier_dataset")

    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('fwd_model_dir', type=pathlib.Path, help='forward model', nargs="+")
    parser.add_argument('--total-take', type=int, help="will be split up between train/test/val")
    parser.add_argument('--start-at', type=int, help='start at this example in the input dynamic dataste')
    parser.add_argument('--stop-at', type=int, help='start at this example in the input dynamic dataste')
    parser.add_argument('out_dir', type=pathlib.Path, help='out dir')

    args = parser.parse_args()

    labeling_params = json.load(args.labeling_params.open("r"))
    dynamics_hparams = json.load((args.dataset_dir / 'hparams.json').open('r'))
    fwd_models, _ = model_utils.load_generic_model(args.fwd_model_dir)

    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')

    dataset = DynamicsDataset([args.dataset_dir])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # success = mkdir_and_ask(args.out_dir, parents=True)
    # if not success:
    #     print(Fore.RED + "Aborting" + Fore.RESET)
    #     return

    new_hparams_filename = args.out_dir / 'hparams.json'
    classifier_dataset_hparams = dynamics_hparams
    if len(args.fwd_model_dir) > 1:
        using_ensemble = True
        fwd_model_dir = [str(d) for d in args.fwd_model_dir]
    else:
        using_ensemble = False
        fwd_model_dir = str(args.fwd_model_dir[0])
    classifier_dataset_hparams['dataset_dir'] = str(args.dataset_dir)
    classifier_dataset_hparams['fwd_model_dir'] = fwd_model_dir
    classifier_dataset_hparams['fwd_model_hparams'] = fwd_models.hparams
    classifier_dataset_hparams['using_ensemble'] = using_ensemble
    classifier_dataset_hparams['labeling_params'] = labeling_params
    classifier_dataset_hparams['state_keys'] = fwd_models.state_keys
    classifier_dataset_hparams['action_keys'] = fwd_models.action_keys
    classifier_dataset_hparams['start-at'] = args.start_at
    classifier_dataset_hparams['stop-at'] = args.stop_at
    json.dump(classifier_dataset_hparams, new_hparams_filename.open("w"), indent=2)

    val_split = int(args.total_take * DEFAULT_VAL_SPLIT) if args.total_take is not None else None
    test_split = int(args.total_take * DEFAULT_TEST_SPLIT) if args.total_take is not None else None
    train_split = args.total_take - val_split - test_split if args.total_take is not None else None
    take_split = {
        'test': test_split,
        'val': val_split,
        'train': train_split
    }

    t0 = perf_counter()
    total_count = 0
    for mode in ['train', 'val', 'test']:
        tf_dataset = dataset.get_datasets(mode=mode, take=take_split[mode])

        full_output_directory = args.out_dir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        for out_example in generate_classifier_examples(fwd_models, tf_dataset, dataset, labeling_params):
            for batch_idx in range(out_example['traj_idx'].shape[0]):
                out_example_b = index_dict_of_batched_vectors_tf(out_example, batch_idx)
                features = {k: float_tensor_to_bytes_feature(v) for k, v in out_example_b.items()}

                example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                example = example_proto.SerializeToString()
                record_filename = "example_{:09d}.tfrecords".format(total_count)
                full_filename = full_output_directory / record_filename
                with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
                    writer.write(example)
                total_count += 1
                print(f"Examples: {total_count:10d}, Time: {perf_counter() - t0:.3f}")


if __name__ == '__main__':
    main()
