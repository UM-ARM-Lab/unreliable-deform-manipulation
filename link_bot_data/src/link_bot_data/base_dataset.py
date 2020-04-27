#!/usr/bin/env python
import json
import pathlib
from typing import List, Optional

import tensorflow as tf
from colorama import Fore

from link_bot_data.link_bot_dataset_utils import parse_and_deserialize

DEFAULT_VAL_SPLIT = 0.125
DEFAULT_TEST_SPLIT = 0.125


class BaseDataset:

    def __init__(self, dataset_dirs: List[pathlib.Path]):
        self.dataset_dirs = dataset_dirs
        self.hparams = {}
        for dataset_dir in dataset_dirs:
            dataset_hparams_filename = dataset_dir / 'hparams.json'

            # to merge dataset hparams
            hparams = json.load(dataset_hparams_filename.open('r'))
            for k, v in hparams.items():
                if k not in self.hparams:
                    self.hparams[k] = v
                elif self.hparams[k] == v:
                    pass
                elif k == 'sequence_length':
                    # always use the minimum of different sequence lengths
                    self.hparams[k] = min(self.hparams[k], hparams[k])
                else:
                    msg = "Datasets have differing values for the hparam {}, using value {}".format(k, self.hparams[k])
                    print(Fore.RED + msg + Fore.RESET)

        self.desired_sequence_length = None
        self.max_sequence_length = self.hparams['sequence_length']

    def get_datasets(self,
                     mode: str,
                     sequence_length: Optional[int] = None,
                     n_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                     do_not_process: bool = False,
                     shard: Optional[int] = None,
                     take: Optional[int] = None,
                     ) -> tf.data.Dataset:
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

        self.desired_sequence_length = sequence_length if sequence_length is not None else self.max_sequence_length
        all_filenames = sorted(all_filenames)
        return self.get_datasets_from_records(all_filenames,
                                              n_parallel_calls=n_parallel_calls,
                                              do_not_process=do_not_process,
                                              shard=shard,
                                              take=take)

    def get_datasets_from_records(self,
                                  records: List[str],
                                  n_parallel_calls: Optional[int] = None,
                                  do_not_process: Optional[bool] = False,
                                  shard: Optional[int] = None,
                                  take: Optional[int] = None,
                                  ) -> tf.data.Dataset:
        dataset = tf.data.TFRecordDataset(records, buffer_size=1 * 1024 * 1024, compression_type='ZLIB')

        # Given the member lists of states, actions, and constants set in the constructor, create a dict for parsing a feature
        features_description = self.make_features_description()
        dataset = parse_and_deserialize(dataset, feature_description=features_description, n_parallel_calls=n_parallel_calls)

        if take is not None:
            dataset = dataset.take(take)

        if shard is not None:
            dataset = dataset.shard(shard)

        if not do_not_process:
            dataset = self.post_process(dataset, n_parallel_calls)

        return dataset

    def make_features_description(self):
        raise NotImplementedError()

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        # No-Op
        return dataset


def make_name_singular(feature_name):
    if "/" in feature_name:
        base_feature_name, postfix = feature_name.split("/")
        if not base_feature_name.endswith("_s"):
            raise ValueError("time-indexed feature name {} doesn't end with _s ".format(base_feature_name))
        base_feature_name_singular = base_feature_name[:-2]
        feature_name_singular = "{}/{}".format(base_feature_name_singular, postfix)
        next_feature_name_singular = "{}_next/{}".format(base_feature_name_singular, postfix)
    else:
        base_feature_name = feature_name
        if not base_feature_name.endswith("_s"):
            raise ValueError("time-indexed feature name {} doesn't end with _s ".format(base_feature_name))
        base_feature_name_singular = base_feature_name[:-2]
        feature_name_singular = base_feature_name_singular
        next_feature_name_singular = "{}_next".format(base_feature_name_singular)
    return feature_name_singular, next_feature_name_singular
