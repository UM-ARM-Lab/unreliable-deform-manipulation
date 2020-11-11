#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.args import my_formatter
from moonshine.moonshine_utils import numpify


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    rospy.init_node("modify_dynamics_dataset")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        rope_vec = numpify(example.pop('rope'))
        rope_points = rope_vec.reshape([10, 25, 3])
        rope_points_rev = rope_points[:, ::-1]
        example['rope'] = rope_points_rev.reshape([10, -1])
        yield example

    hparams_update = {}

    dataset = DynamicsDatasetLoader([args.dataset_dir])
    modify_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   process_example=_process_example,
                   hparams_update=hparams_update)


if __name__ == '__main__':
    main()
