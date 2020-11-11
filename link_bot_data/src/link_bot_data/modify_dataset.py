#!/usr/bin/env python
import pathlib
from typing import Callable, Optional, Dict

import hjson
import progressbar

from arc_utilities import algorithms
from link_bot_data.base_dataset import BaseDatasetLoader
from link_bot_data.dataset_utils import tf_write_example


def modify_hparams(in_dir: pathlib.Path, out_dir: pathlib.Path, update: Optional[Dict] = None):
    if update is None:
        update = {}
    out_dir.mkdir(exist_ok=True, parents=False)
    with (in_dir / 'hparams.hjson').open("r") as in_f:
        in_hparams_str = in_f.read()
    in_hparams = hjson.loads(in_hparams_str)

    out_hparams = in_hparams
    algorithms.update(out_hparams, update)
    out_hparams_str = hjson.dumps(out_hparams)
    with (out_dir / 'hparams.hjson').open("w") as out_f:
        out_f.write(out_hparams_str)


def modify_dataset(dataset_dir: pathlib.Path,
                   dataset: BaseDatasetLoader,
                   outdir: pathlib.Path,
                   process_example: Callable,
                   hparams_update: Optional[Dict] = None):
    if hparams_update is None:
        hparams_update = {}

    # hparams
    modify_hparams(dataset_dir, outdir, hparams_update)

    # tfrecords
    total_count = 0
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    for mode in ['train', 'test', 'val']:
        tf_dataset = dataset.get_datasets(mode=mode)
        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        for i, example in enumerate(tf_dataset):
            for out_example in process_example(dataset, example):
                for k in dataset.scenario_metadata:
                    out_example.pop(k)
                tf_write_example(full_output_directory, out_example, total_count)
                total_count += 1
                bar.update(total_count)
