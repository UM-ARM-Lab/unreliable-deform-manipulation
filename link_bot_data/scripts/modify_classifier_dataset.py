#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama

import rospy
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import add_predicted
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.args import my_formatter


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('suffix', type=str, help='string added to the new dataset name')

    args = parser.parse_args()

    rospy.init_node("modify_dynamics_dataset")

    outdir = args.dataset_dir.parent / f"{args.dataset_dir.name}+{args.suffix}"

    def _process_example(dataset: ClassifierDatasetLoader, example: Dict):
        example['left_gripper'] = example.pop('gripper1')
        example['right_gripper'] = example.pop('gripper2')
        example['left_gripper_position'] = example.pop('gripper1_position')
        example['right_gripper_position'] = example.pop('gripper2_position')
        example['rope'] = example.pop('link_bot')
        example[add_predicted('left_gripper')] = example.pop(add_predicted('gripper1'))
        example[add_predicted('right_gripper')] = example.pop(add_predicted('gripper2'))
        example[add_predicted('rope')] = example.pop(add_predicted('link_bot'))
        yield example

    hparams_update = {}

    dataset = ClassifierDatasetLoader([args.dataset_dir], use_gt_rope=False, load_true_states=True)
    modify_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   process_example=_process_example,
                   hparams_update=hparams_update)


if __name__ == '__main__':
    main()
