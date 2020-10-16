#!/usr/bin/env python
import pathlib
from typing import Callable, Optional, Dict

import hjson
import tensorflow as tf

from arc_utilities import algorithms
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature


def modify_hparams(in_dir: pathlib.Path, out_dir: pathlib.Path, update: Optional[Dict] = None):
    if update is None:
        update = {}
    out_dir.mkdir(exist_ok=True, parents=False)
    with (in_dir / 'hparams.json').open("r") as in_f:
        in_hparams_str = in_f.read()
    in_hparams = hjson.loads(in_hparams_str)

    out_hparams = in_hparams
    algorithms.update(out_hparams, update)
    out_hparams_str = hjson.dumps(out_hparams)
    with (out_dir / 'hparams.json').open("w") as out_f:
        out_f.write(out_hparams_str)


def modify_dynamics_dataset(dataset_dir: pathlib.Path,
                            outdir: pathlib.Path,
                            process_example: Callable,
                            hparams_update: Optional[Dict] = None):
    if hparams_update is None:
        hparams_update = {}
    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')

    # load the dataset
    dataset = DynamicsDataset([dataset_dir])

    # hparams
    modify_hparams(dataset_dir, outdir, hparams_update)

    # tfrecords
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
