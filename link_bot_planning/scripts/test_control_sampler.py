#!/usr/bin/env python
import argparse
import pathlib

from link_bot_planning import model_utils
from link_bot_planning.shooting_directed_control_sampler import ShootingDirectedControlSamplerInternal
from link_bot_pycommon.args import my_formatter


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='gp')

    args = parser.parse_args()

    fwd_model = model_utils.load_generic_model(args.fwd_model_dir, args.fwd_model_type)

    control_sampler = ShootingDirectedControlSamplerInternal(n_state=6, n_local_sdf=100 * 100)


if __name__ == '__main__':
    main()
