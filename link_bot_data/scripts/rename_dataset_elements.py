#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import numpy as np

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dynamics_dataset import modify_dynamics_dataset
from link_bot_pycommon.args import my_formatter


def main():
    colorama.init(autoreset=True)
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("rename_grippers")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+renamed')

    # UPDATE THIS TOO!
    hparams_update = {
        # 'states_description':               {'gt_rope': 75, },
        # 'observation_features_description': {'gt_rope': 75, },
    }

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        new_rope = example.pop("cdcpd")
        gt_rope = example.pop("rope")
        example['rope'] = new_rope
        example['gt_rope'] = gt_rope
        # example['left_gripper'] = example.pop("gripper1")
        # example['right_gripper'] = example.pop("gripper2")
        # example['left_gripper_position'] = example.pop("gripper1_position")
        # example['right_gripper_position'] = example.pop("gripper2_position")
        # example['rope'] = example.pop("link_bot")
        example.pop("joint_names")
        yield example

    modify_dynamics_dataset(args.dataset_dir, outdir, _process_example, hparams_update=hparams_update)


if __name__ == '__main__':
    main()
