#!/usr/bin/env python
from __future__ import print_function, division

import argparse

import numpy as np
import tensorflow

from link_bot_data import base_collect_dynamics_data
from link_bot_pycommon.args import my_formatter
from victor import victor_services

opts = tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
conf = tensorflow.compat.v1.ConfigProto(gpu_options=opts)
tensorflow.compat.v1.enable_eager_execution(config=conf)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=5000)
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.DEBUG)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("trajs", type=int, help='how many trajectories to collect')
    parser.add_argument("outdir")
    parser.add_argument('--dt', type=float, default=1.00, help='seconds to execute each delta position action')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument('--env-w', type=float, default=6.0, help='full env w')
    parser.add_argument('--env-h', type=float, default=6.0, help='full env h')
    parser.add_argument('--goal-env-w', type=float, default=2.2, help='goal env w')
    parser.add_argument('--goal-env-h', type=float, default=2.2, help='goal env h')
    parser.add_argument('--local_env-cols', type=int, default=50, help='local env')
    parser.add_argument('--local_env-rows', type=int, default=50, help='local env')
    parser.add_argument("--steps-per-traj", type=int, default=100, help='steps per traj')
    parser.add_argument("--start-idx-offset", type=int, default=0, help='offset TFRecord file names')
    parser.add_argument("--move-objects-every-n", type=int, default=16, help='rearrange objects every n trajectories')
    parser.add_argument("--no-obstacles", action='store_true', help='do not move obstacles')
    parser.add_argument("--trajs-per-file", type=int, default=128, help='trajs per file')
    parser.add_argument("--seed", '-s', type=int, help='seed')
    parser.add_argument("--real-time-rate", type=float, default=0, help='number of times real time')
    parser.add_argument("--max-step-size", type=float, default=0.01, help='seconds per physics step')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    base_collect_dynamics_data.generate(victor_services, args)


if __name__ == '__main__':
    main()
