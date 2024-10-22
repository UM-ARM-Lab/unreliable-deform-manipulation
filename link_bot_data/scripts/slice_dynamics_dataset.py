#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dataset import modify_dataset
from link_bot_pycommon.args import my_formatter
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    colorama.init(autoreset=True)
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('desired_sequence_length', type=int, help='desired seqeuence length')

    args = parser.parse_args()

    rospy.init_node("slice_dataset")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + f'+L{args.desired_sequence_length}')

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        out_examples = dataset.split_into_sequences(example, args.desired_sequence_length)
        for out_example in out_examples:
            out_example['time_idx'] = tf.range(0, args.desired_sequence_length, dtype=tf.float32)
            yield out_example

    hparams_update = {
        'data_collection_params': {
            'steps_per_traj': args.desired_sequence_length
        }
    }

    dataset = DynamicsDatasetLoader([args.dataset_dir])
    modify_dataset(dataset_dir=args.dataset_dir,
                   dataset=dataset,
                   outdir=outdir,
                   process_example=_process_example,
                   hparams_update=hparams_update)


if __name__ == '__main__':
    main()
