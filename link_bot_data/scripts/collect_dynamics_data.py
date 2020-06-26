#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import json
import pathlib

import numpy as np

import rospy
from link_bot_data import base_collect_dynamics_data
from link_bot_gazebo import gazebo_services
from link_bot_pycommon.args import my_formatter
from moonshine.gpu_config import limit_gpu_mem
from victor import victor_services

limit_gpu_mem(0.1)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=5000)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("service_provider", choices=['victor', 'gazebo'], default='gazebo', help='victor or gazebo')
    parser.add_argument("scenario", choices=['link_bot', 'dual_floating_gripper_rope', 'dual'], help='scenario')
    parser.add_argument("collect_dynamics_params", type=pathlib.Path, help="json file with envrionment parameters")
    parser.add_argument("trajs", type=int, help='how many trajectories to collect')
    parser.add_argument("outdir")
    parser.add_argument("--start-idx-offset", type=int, default=0, help='offset TFRecord file names')
    parser.add_argument("--seed", '-s', type=int, help='seed')
    parser.add_argument("--real-time-rate", type=float, default=0, help='number of times real time')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    with args.collect_dynamics_params.open("r") as f:
        collect_dynamics_params = json.load(f)

    # Start Services
    if args.service_provider == 'victor':
        rospy.set_param('service_provider', 'victor')
        service_provider = victor_services.VictorServices()
    else:
        rospy.set_param('service_provider', 'gazebo')
        service_provider = gazebo_services.GazeboServices()

    base_collect_dynamics_data.generate(service_provider, collect_dynamics_params, args)


if __name__ == '__main__':
    main()
