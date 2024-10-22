#!/usr/bin/env python
import csv
import pathlib
from typing import List, Optional, Dict, Callable, Any

import hjson
import numpy as np
import progressbar
import tensorflow as tf
from colorama import Fore

from link_bot_data.dataset_utils import parse_and_deserialize

DEFAULT_VAL_SPLIT = 0.125
DEFAULT_TEST_SPLIT = 0.125

SORT_FILE_NAME = 'sort_order.csv'

widgets = [
    progressbar.Percentage(), ' ',
    progressbar.Counter(), ' ',
    progressbar.Bar(),
    ' (', progressbar.AdaptiveETA(), ') ',
]


class SizedTFDataset:

    def __init__(self, dataset: tf.data.Dataset, records: List, size: Optional[int] = None):
        self.dataset = dataset
        self.records = records
        if size is None:
            self.size = len(self.records)
        else:
            self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.dataset.__iter__()

    def __repr__(self):
        if len(self.records) > 0:
            dir = pathlib.Path(self.records[0]).parent.as_posix()
        else:
            dir = "??"
        return f"Dataset: {dir}, size={self.size}"

    def batch(self, batch_size: int, *args, **kwargs):
        dataset_batched = self.dataset.batch(*args, batch_size=batch_size, **kwargs)
        return SizedTFDataset(dataset_batched, self.records, size=int(self.size / batch_size))

    def map(self, function: Callable):
        dataset_mapped = self.dataset.map(function)
        return SizedTFDataset(dataset_mapped, self.records)

    def filter(self, function: Callable):
        dataset_filter = self.dataset.filter(function)
        return SizedTFDataset(dataset_filter, self.records, size=None)

    def repeat(self, count: Optional[int] = None):
        dataset = self.dataset.repeat(count)
        return SizedTFDataset(dataset, self.records, size=None)

    def shuffle(self, buffer_size: int, seed: Optional[int] = None, reshuffle_each_iteration: bool = False):
        dataset = self.dataset.shuffle(buffer_size, seed, reshuffle_each_iteration)
        return SizedTFDataset(dataset, self.records)

    def prefetch(self, buffer_size: Any = tf.data.experimental.AUTOTUNE):
        dataset = self.dataset.prefetch(buffer_size)
        return SizedTFDataset(dataset, self.records)

    def take(self, count: int):
        dataset = self.dataset.take(count)
        return SizedTFDataset(dataset, self.records, size=count)


class BaseDatasetLoader:

    def __init__(self, dataset_dirs: List[pathlib.Path]):
        self.dataset_dirs = dataset_dirs
        self.hparams = {}
        for dataset_dir in dataset_dirs:
            dataset_hparams_filename = dataset_dir / 'hparams.json'
            if not dataset_hparams_filename.exists():
                dataset_hparams_filename = dataset_dir / 'hparams.hjson'

            # to merge dataset hparams
            hparams = hjson.load(dataset_hparams_filename.open('r'))
            for k, v in hparams.items():
                if k not in self.hparams:
                    self.hparams[k] = v
                elif self.hparams[k] == v:
                    pass
                else:
                    msg = "Datasets have differing values for the hparam {}, using value {}".format(k, self.hparams[k])
                    print(Fore.RED + msg + Fore.RESET)

        self.scenario_metadata = self.hparams.get('scenario_metadata', {})

    def get_datasets(self,
                     mode: str,
                     n_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                     do_not_process: bool = False,
                     shard: Optional[int] = None,
                     take: Optional[int] = None,
                     shuffle_files: Optional[bool] = False,
                     sort: Optional[bool] = False,
                     **kwargs):
        all_filenames = self.get_record_filenames(mode, sort=sort)
        return self.get_datasets_from_records(all_filenames,
                                              n_parallel_calls=n_parallel_calls,
                                              do_not_process=do_not_process,
                                              shard=shard,
                                              shuffle_files=shuffle_files,
                                              take=take,
                                              **kwargs)

    def get_record_filenames(self, mode: str, sort: Optional[bool] = False):
        if mode == 'all':
            train_filenames = []
            test_filenames = []
            val_filenames = []
            for dataset_dir in self.dataset_dirs:
                train_filenames.extend(str(filename) for filename in dataset_dir.glob("{}/*.tfrecords".format('train')))
                test_filenames.extend(str(filename) for filename in dataset_dir.glob("{}/*.tfrecords".format('test')))
                val_filenames.extend(str(filename) for filename in dataset_dir.glob("{}/*.tfrecords".format('val')))

            all_filenames = train_filenames
            all_filenames.extend(test_filenames)
            all_filenames.extend(val_filenames)
        else:
            all_filenames = []
            for dataset_dir in self.dataset_dirs:
                all_filenames.extend(str(filename) for filename in (dataset_dir / mode).glob("*.tfrecords"))

        # Initially sort lexicographically, which is intended to make things read in the order they were written
        all_filenames = sorted(all_filenames)

        # Sorting
        if len(self.dataset_dirs) > 1 and sort:
            raise NotImplementedError("Don't know how to make a sorted multi-dataset")

        if sort:
            sorted_filename = self.dataset_dirs[0] / SORT_FILE_NAME
            with open(sorted_filename, "r") as sorted_file:
                reader = csv.reader(sorted_file)
                sort_info_filenames = [row[0] for row in reader]

                # sort all_filenames based on the order of sort_info
                def _sort_key(filename):
                    return sort_info_filenames.index(filename)

                all_filenames = sorted(all_filenames, key=_sort_key)

        return all_filenames

    def get_datasets_from_records(self,
                                  records: List[str],
                                  n_parallel_calls: Optional[int] = None,
                                  do_not_process: Optional[bool] = False,
                                  shard: Optional[int] = None,
                                  take: Optional[int] = None,
                                  shuffle_files: Optional[bool] = False,
                                  **kwargs,
                                  ):
        if shuffle_files:
            print("Shuffling records")
            shuffle_rng = np.random.RandomState(0)
            shuffle_rng.shuffle(records)

        dataset = tf.data.TFRecordDataset(records, buffer_size=1 * 1024 * 1024, compression_type='ZLIB')

        # Given the member lists of states, actions, and constants set in the constructor, create a dict for parsing a feature
        features_description = self.make_features_description()
        dataset = parse_and_deserialize(dataset, features_description, n_parallel_calls)

        if take is not None:
            dataset = dataset.take(take)

        if shard is not None:
            dataset = dataset.shard(shard)

        if not do_not_process:
            dataset = self.post_process(dataset, n_parallel_calls)

        sized_tf_dataset = SizedTFDataset(dataset, records)
        return sized_tf_dataset

    def make_features_description(self):
        if 'has_tfrecord_path' in self.hparams and self.hparams['has_tfrecord_path']:
            return {'tfrecord_path': tf.io.FixedLenFeature([], tf.string)}
        else:
            return {}

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        scenario_metadata = self.scenario_metadata

        def _add_scenario_metadata(example: Dict):
            example.update(scenario_metadata)
            return example

        dataset = dataset.map(_add_scenario_metadata)
        return dataset
