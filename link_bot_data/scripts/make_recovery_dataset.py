#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import tensorflow as tf
from colorama import Fore

import rospy
from arc_utilities.filesystem_utils import mkdir_and_ask
from link_bot_data.recovery_dataset_utils import make_recovery_dataset
from link_bot_pycommon.args import my_formatter
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(7.5)


def main():
    colorama.init(autoreset=True)

    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('fwd_model_dir', type=pathlib.Path, help='forward model', nargs="+")
    parser.add_argument('classifier_model_dir', type=pathlib.Path)
    parser.add_argument('out_dir', type=pathlib.Path, help='out dir')
    parser.add_argument('--start-at', type=int, help='start at this example in the input dynamic dataste')
    parser.add_argument('--stop-at', type=int, help='start at this example in the input dynamic dataste')
    parser.add_argument('--max-examples-per-record', type=int, default=128, help="examples per file")
    parser.add_argument('--batch-size', type=int, help="batch size", default=2)

    args = parser.parse_args()

    success = mkdir_and_ask(args.out_dir, parents=True)
    if not success:
        print(Fore.RED + "Aborting" + Fore.RESET)
        return

    rospy.init_node("make_recovery_dataset")

    make_recovery_dataset(dataset_dir=args.dataset_dir,
                          fwd_model_dir=args.fwd_model_dir,
                          classifier_model_dir=args.classifier_model_dir,
                          labeling_params=args.labeling_params,
                          outdir=args.out_dir,
                          batch_size=args.batch_size,
                          start_at=args.start_at,
                          stop_at=args.stop_at)


if __name__ == '__main__':
    main()
