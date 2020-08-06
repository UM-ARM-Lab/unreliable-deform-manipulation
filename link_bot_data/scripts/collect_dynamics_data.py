#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import json
import pathlib

import numpy as np

from link_bot_data import base_collect_dynamics_data
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_service_provider import get_service_provider
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=5000)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("service_provider", choices=['victor', 'gazebo'], default='gazebo', help='victor or gazebo')
    parser.add_argument("scenario", choices=['dragging', 'dual_floating', 'dual_arm'], help='scenario')
    parser.add_argument("collect_dynamics_params", type=pathlib.Path, help="json file with envrionment parameters")
    parser.add_argument("n_trajs", type=int, help='how many trajectories to collect')
    parser.add_argument("nickname")
    parser.add_argument("--seed", '-s', type=int, help='seed')
    parser.add_argument("--real-time-rate", type=float, default=0, help='number of times real time')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    with args.collect_dynamics_params.open("r") as f:
        collect_dynamics_params = json.load(f)

    service_provider = get_service_provider(args.service_provider)

    data_collector = base_collect_dynamics_data.DataCollector(scenario_name=args.scenario,
                                                              service_provider=service_provider,
                                                              params=collect_dynamics_params,
                                                              seed=args.seed,
                                                              verbose=args.verbose)
    files_dataset = data_collector.collect_data(n_trajs=args.n_trajs, nickname=args.nickname)
    files_dataset.split()


if __name__ == '__main__':
    main()
