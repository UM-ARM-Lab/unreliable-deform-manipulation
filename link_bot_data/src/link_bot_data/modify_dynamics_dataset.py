#!/usr/bin/env python
import pathlib
import shutil
from typing import Callable

import tensorflow as tf

from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature


def modify_dynamics_dataset(dataset_dir: pathlib.Path, outdir: pathlib.Path, process_example: Callable):
    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')

    # load the dataset
    dataset = DynamicsDataset([dataset_dir])

    outdir.mkdir(exist_ok=True, parents=False)
    in_hparams = dataset_dir / 'hparams.json'
    out_hparams = outdir / 'hparams.json'
    shutil.copy(in_hparams, out_hparams)

    total_count = 0
    for mode in ['train', 'test', 'val']:
        tf_dataset = dataset.get_datasets(mode=mode)
        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        for i, example in enumerate(tf_dataset):
            for out_example in process_example(dataset, example):
                features = {k: float_tensor_to_bytes_feature(v) for k, v in out_example.items()}
                example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                example_str = example_proto.SerializeToString()
                record_filename = "example_{:09d}.tfrecords".format(total_count)
                full_filename = full_output_directory / record_filename
                with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
                    writer.write(example_str)
                total_count += 1


if __name__ == '__main__':
    main()
