#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt

from link_bot_classifiers import none_classifier
from link_bot_data.visualization import plot_rope_configuration
from link_bot_gazebo import gazebo_utils
from link_bot_planning import model_utils
from link_bot_planning.params import LocalEnvParams
from link_bot_planning.shooting_directed_control_sampler import ShootingDirectedControlSamplerInternal
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_pycommon import make_random_rope_configuration


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='gp')
    parser.add_argument("--n-samples", type=int, help='number of actions to sample', default=10)

    args = parser.parse_args()

    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir, args.fwd_model_type)
    classifier_model = none_classifier.NoneClassifier()

    services = gazebo_utils.GazeboServices()

    # NOTE: these params don't matter at the moment, but they may soon
    local_env_params = LocalEnvParams(h_rows=100, w_cols=100, res=0.03)

    control_sampler = ShootingDirectedControlSamplerInternal(fwd_model=fwd_model,
                                                             classifier_model=classifier_model,
                                                             services=services,
                                                             local_env_params=local_env_params,
                                                             max_v=0.15,
                                                             n_samples=args.n_samples,
                                                             n_state=6,
                                                             n_local_env=100 * 100)

    initial_state = make_random_rope_configuration(extent=[-2.5, 2.5, -2.5, 2.5])
    target_state = make_random_rope_configuration(extent=[-2.5, 2.5, -2.5, 2.5])

    reached_state, u, local_env = control_sampler.sampleTo(initial_state, target_state)

    plt.figure()
    ax = plt.gca()
    plot_rope_configuration(ax, initial_state, c='r', label='initial')
    plot_rope_configuration(ax, target_state, c='g', label='target')
    plot_rope_configuration(ax, reached_state, c='b', label='reached')


if __name__ == '__main__':
    main()
