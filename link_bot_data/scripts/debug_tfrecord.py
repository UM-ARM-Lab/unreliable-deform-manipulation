#!/usr/bin/env python
import json
from colorama import Fore
import pathlib

import tensorflow as tf
from google.protobuf.json_format import MessageToDict
import argparse

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=pathlib.Path)
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')

    args = parser.parse_args()

    dataset_hparams_filename = args.input_dir / 'hparams.json'
    hparams = json.load(open(str(dataset_hparams_filename), 'r'))

    filenames = [str(filename) for filename in args.input_dir.glob("{}/*.tfrecords".format(args.mode))]
    options = tf.python_io.TFRecordOptions(compression_type=hparams['compression_type'])
    for filename in filenames:
        example = next(tf.python_io.tf_record_iterator(filename, options=options))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']

        for feature_name, feature_value in feature.items():
            type_name = list(feature_value.keys())[0]
            feature_value = feature_value[type_name]
            if 'value' in feature_value.keys():
                feature_value = feature_value['value']
                print(feature_name, len(feature_value))
            else:
                print(Fore.RED + "Empty feature: {}, {}".format(feature_name, feature_value) + Fore.RESET)


if __name__ == '__main__':
    main()
