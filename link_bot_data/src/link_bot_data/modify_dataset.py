#!/usr/bin/env python
import pathlib
from typing import Callable, Optional, Dict

import hjson
import progressbar
from colorama import Fore

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


def sort_dataset_mode(mode: str, dataset: BaseDatasetLoader, get_value: Callable):
    tf_dataset = dataset.get_datasets(mode=mode)

    values_and_indices = []
    for i, example in enumerate(tf_dataset):
        value = get_value(dataset, example)
        values_and_indices.append((value, i))

    sorted_values_and_indices = sorted(values_and_indices)
    sorted_indices = [i for v, i in sorted_values_and_indices]
    return sorted_indices


def sort_dataset(dataset_dir, dataset: BaseDatasetLoader, get_value: Callable):
    for mode in ['train', 'test', 'val']:
        sorted_indices = sort_dataset_mode(mode, dataset, get_value)
        sort_filename = dataset_dir / mode / 'sort_order.txt'
        # https://stackoverflow.com/questions/13730107/writelines-writes-lines-without-newline-just-fills-the-file
        sorted_indices_str = [f"{i}\n" for i in sorted_indices]

        with open(sort_filename, 'w') as sort_file:
            sort_file.writelines(sorted_indices_str)


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
    with progressbar.ProgressBar(max_value=progressbar.UnknownLength) as bar:
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
    print(Fore.GREEN + f"Modified {total_count} examples")
