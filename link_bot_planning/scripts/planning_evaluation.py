#!/usr/bin/env python
import rospy
import traceback
import numpy as np
import json
import ompl.util as ou
from colorama import Fore
import argparse
import pathlib

from link_bot_pycommon.args import my_formatter, int_range_arg
from moonshine.gpu_config import limit_gpu_mem
from link_bot_data.link_bot_dataset_utils import data_directory
from link_bot_planning.planning_evaluation import evaluate_planning_method

limit_gpu_mem(7.5)


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=250)
    ou.setLogLevel(ou.LOG_ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('planners_params', type=pathlib.Path, nargs='+',
                        help='json file(s) describing what should be compared')
    parser.add_argument("trials", type=int_range_arg, default="0-50")
    parser.add_argument("nickname", type=str, help='output will be in results/$nickname-compare-$time')
    parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    parser.add_argument("--no-execution", action="store_true", help='no execution')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--record', action='store_true', help='record')
    parser.add_argument('--skip-on-exception', action='store_true', help='skip method if exception is raise')

    args = parser.parse_args()

    root = pathlib.Path('results') / "{}-compare".format(args.nickname)
    common_output_directory = data_directory(root)
    common_output_directory = pathlib.Path(common_output_directory)
    print(Fore.CYAN + "common output directory: {}".format(common_output_directory) + Fore.RESET)
    if not common_output_directory.is_dir():
        print(Fore.YELLOW + "Creating output directory: {}".format(common_output_directory) + Fore.RESET)
        common_output_directory.mkdir(parents=True)

    rospy.init_node("planning_evaluation")

    planners_params = [(json.load(p_params_name.open("r")), p_params_name) for p_params_name in args.planners_params]
    for comparison_idx, (planner_params, p_params_name) in enumerate(planners_params):
        if args.skip_on_exception:
            try:
                evaluate_planning_method(args, comparison_idx, planner_params, p_params_name, common_output_directory)
            except Exception as e:
                traceback.print_exc()
                print(e)
        else:
            evaluate_planning_method(args, comparison_idx, planner_params, p_params_name, common_output_directory)
        print(f"Results written to {common_output_directory}")


if __name__ == '__main__':
    main()
