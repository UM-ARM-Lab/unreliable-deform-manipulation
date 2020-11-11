#!/usr/bin/env python
import argparse
import pathlib
from typing import Type

import colorama
import hjson
import numpy as np

import moveit_commander
import rospy
from link_bot_data.base_collect_dynamics_data import TfDataCollector, H5DataCollector
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_service_provider import get_service_provider
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def get_data_collector_class(save_format: str) -> Type:
    if save_format == 'h5':
        return H5DataCollector
    elif save_format == 'tfrecord':
        return TfDataCollector
    else:
        raise NotImplementedError(f"unsupported save_format {save_format}")


def main():
    colorama.init(autoreset=True)

    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=5000)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("service_provider", choices=['victor', 'gazebo'], default='gazebo', help='victor or gazebo')
    parser.add_argument("scenario", type=str, help='scenario')
    parser.add_argument("robot_namespace", type=str, help='robot_namespace')
    parser.add_argument("collect_dynamics_params", type=pathlib.Path, help="json file with envrionment parameters")
    parser.add_argument("n_trajs", type=int, help='how many trajectories to collect')
    parser.add_argument("nickname")
    parser.add_argument("--seed", '-s', type=int, help='seed')
    parser.add_argument("--real-time-rate", type=float, default=0, help='number of times real time')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--save-format', choices=['h5', 'tfrecord'], default='tfrecord')

    args = parser.parse_args()

    moveit_commander.roscpp_initialize(args=[])
    rospy.init_node('collect_dynamics_data')

    with args.collect_dynamics_params.open("r") as f:
        collect_dynamics_params = hjson.load(f)

    service_provider = get_service_provider(args.service_provider)

    DataCollectorClass = get_data_collector_class(args.save_format)
    data_collector = DataCollectorClass(scenario_name=args.scenario,
                                        service_provider=service_provider,
                                        params=collect_dynamics_params,
                                        seed=args.seed,
                                        verbose=args.verbose)
    files_dataset = data_collector.collect_data(n_trajs=args.n_trajs, nickname=args.nickname,
                                                robot_namespace=args.robot_namespace)
    files_dataset.split()


if __name__ == '__main__':
    main()
