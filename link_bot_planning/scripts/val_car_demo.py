#!/usr/bin/env python
import colorama
import rospy
import traceback
import numpy as np
import json
import ompl.util as ou
from colorama import Fore
import argparse
import pathlib

from link_bot_pycommon.args import my_formatter
from moonshine.gpu_config import limit_gpu_mem
from link_bot_data.link_bot_dataset_utils import data_directory
from link_bot_planning.planning_evaluation import evaluate_planning_method

limit_gpu_mem(7.5)


def main():
    colorama.init(autoreset=True)
    np.set_printoptions(precision=6, suppress=True, linewidth=250)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('planners_params', type=pathlib.Path, nargs='+',
                        help='json file(s) describing what should be compared')
    parser.add_argument("--nickname", type=str, help='output will be in results/$nickname-compare-$time', required=True)
    parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    parser.add_argument("--no-execution", action="store_true", help='no execution')
    parser.add_argument("--seed", '-s', type=int, default=1)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--record', action='store_true', help='record')

    args = parser.parse_args()

    ou.setLogLevel(ou.LOG_ERROR)
    ou.RNG.setSeed(args.seed)

    print(Fore.CYAN + "Using Seed {}".format(args.seed) + Fore.RESET)

    root = pathlib.Path('results') / "{}-compare".format(args.nickname)
    common_output_directory = data_directory(root)
    common_output_directory = pathlib.Path(common_output_directory)
    print(Fore.CYAN + "common output directory: {}".format(common_output_directory) + Fore.RESET)
    if not common_output_directory.is_dir():
        print(Fore.YELLOW + "Creating output directory: {}".format(common_output_directory) + Fore.RESET)
        common_output_directory.mkdir(parents=True)

    rospy.init_node("planning_evaluation")
    rospy.set_param('service_provider', 'gazebo')

    planners_params = [(json.load(p_params_name.open("r")), p_params_name) for p_params_name in args.planners_params]
    setattr(args, 'n_trials', 1)
    for comparison_idx, (planner_params, p_params_name) in enumerate(planners_params):
        evaluate_planning_method(args, comparison_idx, planner_params, p_params_name, common_output_directory)


if __name__ == '__main__':
    main()
