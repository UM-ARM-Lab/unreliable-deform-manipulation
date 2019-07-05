#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import errno
import os
import pathlib
from shutil import copyfile

import numpy as np

from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")

    args = parser.parse_args()

    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)

    dataset_path = pathlib.Path(args.dataset)
    dataset_directory = dataset_path.parent
    new_dataset_name = dataset_directory.name + "_balanced"

    new_dataset_directory = dataset_directory.parent / new_dataset_name
    new_dataset_filename = new_dataset_directory / "dataset.json"

    try:
        os.makedirs(new_dataset_directory)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(new_dataset_directory):
            pass
        else:
            raise

    # we want to balance based on the "combined" label
    # bigger batches will get this done faster
    new_filename_pairs = []
    for i, (sdf_filename, rope_data_filename) in enumerate(dataset.abs_filename_pairs):
        rope_data_basename = os.path.basename(rope_data_filename)
        sdf_basename = os.path.basename(sdf_filename)
        new_rope_data_filename = dataset_directory.parent / new_dataset_name / rope_data_basename
        new_sdf_filename = dataset_directory.parent / new_dataset_name / sdf_basename
        rope_data = np.load(rope_data_filename)

        new_filename_pairs.append([new_sdf_filename, new_rope_data_filename])

        combined_label = rope_data[LabelType.Combined.name].squeeze()
        positive_count = np.count_nonzero(combined_label)
        negative_count = np.count_nonzero(1 - combined_label)

        negative_indeces = np.argwhere(1 - combined_label).squeeze()
        positive_indeces = np.argwhere(combined_label).squeeze()

        count = min(negative_count, positive_count)
        positive_indeces = np.random.choice(positive_indeces, size=count)
        negative_indeces = np.random.choice(negative_indeces, size=count)

        new_indeces = np.concatenate((negative_indeces, positive_indeces))

        # Save the new rope data
        new_rope_data = {}
        for key, value in rope_data.items():
            # just copy scalars as-is, since they are presumably not actual data but just extra information
            if value.ndim == 0:
                new_rope_data[key] = value
            else:
                new_rope_data[key] = value[new_indeces]
        np.savez(new_rope_data_filename, **new_rope_data)

        # copy the SDF file
        copyfile(sdf_filename, new_sdf_filename)

    dataset = MultiEnvironmentDataset(new_filename_pairs, constraint_label_types=dataset.constraint_label_types,
                                      n_obstacles=dataset.n_obstacles, obstacle_size=dataset.obstacle_size, seed=dataset.seed)
    rounded_num_examples = (dataset.num_examples // 100) * 100
    dataset.slice(rounded_num_examples)
    dataset.save(new_dataset_filename)


if __name__ == '__main__':
    main()
