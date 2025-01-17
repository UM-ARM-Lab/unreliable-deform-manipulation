#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import tensorflow as tf
from colorama import Fore

import rospy
from arc_utilities.filesystem_utils import mkdir_and_ask
from link_bot_data.classifier_dataset_utils import make_classifier_dataset
from link_bot_pycommon.args import my_formatter
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(6)


def main():
    colorama.init(autoreset=True)
    rospy.init_node("make_classifier_dataset")

    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('fwd_model_dir', type=pathlib.Path, help='forward model', nargs="+")
    parser.add_argument('--total-take', type=int, help="will be split up between train/test/val")
    parser.add_argument('--start-at', type=str, help='mode:batch_index, ex train:10')
    parser.add_argument('--stop-at', type=str, help='mode:batch_index, ex train:10')
    parser.add_argument('--batch-size', type=int, help='batch size', default=8)
    parser.add_argument('--yes', '-y', action='store_true')
    parser.add_argument('--use-gt-rope', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('out_dir', type=pathlib.Path, help='out dir')

    args = parser.parse_args()

    outdir = args.out_dir
    success = mkdir_and_ask(outdir, parents=True, yes=args.yes)
    if not success:
        print(Fore.RED + "Aborting" + Fore.RESET)
        return

    rospy.loginfo(Fore.GREEN + f"Writing classifier dataset to {args.out_dir}")
    make_classifier_dataset(dataset_dir=args.dataset_dir,
                            fwd_model_dir=args.fwd_model_dir,
                            labeling_params=args.labeling_params,
                            outdir=outdir,
                            use_gt_rope=args.use_gt_rope,
                            visualize=args.visualize,
                            batch_size=args.batch_size,
                            start_at=args.start_at,
                            stop_at=args.stop_at)

    if __name__ == '__main__':
        main()
