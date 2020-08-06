#!/usr/bin/env python
import argparse
import logging
import pathlib

import rospy
import tensorflow as tf
from colorama import Fore

from link_bot_data.classifier_dataset_utils import make_classifier_dataset
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.filesystem_utils import mkdir_and_ask
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(6)


def main():
    rospy.init_node("make_classifier_dataset")

    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('fwd_model_dir', type=pathlib.Path, help='forward model', nargs="+")
    parser.add_argument('--total-take', type=int, help="will be split up between train/test/val")
    parser.add_argument('--start-at', type=int, help='start at this example in the input dynamic dataste')
    parser.add_argument('--stop-at', type=int, help='start at this example in the input dynamic dataste')
    parser.add_argument('out_dir', type=pathlib.Path, help='out dir')

    args = parser.parse_args()

    success = mkdir_and_ask(args.out_dir, parents=True)
    if not success:
        print(Fore.RED + "Aborting" + Fore.RESET)
        return

    make_classifier_dataset(dataset_dir=args.dataset_dir,
                            fwd_model_dir=args.fwd_model_dir,
                            labeling_params=args.labeling_params,
                            outdir=args.out_dir,
                            start_at=args.start_at,
                            stop_at=args.stop_at)


if __name__ == '__main__':
    main()
