#!/usr/bin/env python
import argparse
import json
import pathlib

import colorama

import rospy
from link_bot_planning.planning_evaluation import planning_evaluation
from link_bot_pycommon.args import my_formatter, int_range_arg
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(7.5)


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('planners_params', type=pathlib.Path, nargs='+',
                        help='json file(s) describing what should be compared')
    parser.add_argument("trials", type=int_range_arg, default="0-50")
    parser.add_argument("nickname", type=str, help='output will be in results/$nickname-compare-$time')
    parser.add_argument("--test-scenes-dir", type=pathlib.Path)
    parser.add_argument("--save-scenes-to", type=pathlib.Path)
    parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    parser.add_argument("--no-execution", action="store_true", help='no execution')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--record', action='store_true', help='record')
    parser.add_argument('--skip-on-exception', action='store_true', help='skip method if exception is raise')

    args = parser.parse_args()

    rospy.init_node("planning_evaluation")

    root = pathlib.Path('results') / f"{args.nickname}-compare"

    planners_params = []
    for planner_params_filename in args.planners_params:
        with planner_params_filename.open('r') as planner_params_file:
            planner_params = json.load(planner_params_file)
        planners_params.append((planner_params_filename.stem, planner_params))

    return planning_evaluation(root=root,
                               planners_params=planners_params,
                               trials=args.trials,
                               skip_on_exception=args.skip_on_exception,
                               verbose=args.verbose,
                               timeout=args.timeout,
                               test_scenes_dir=args.test_scenes_dir,
                               save_test_scenes_dir=args.save_scenes_to,
                               no_execution=args.no_execution,
                               record=args.record)


if __name__ == '__main__':
    main()
