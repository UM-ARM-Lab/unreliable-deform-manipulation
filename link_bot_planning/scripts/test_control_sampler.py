#!/usr/bin/env python
import argparse
import numpy as np
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf

from link_bot_classifiers import none_classifier
from link_bot_data.visualization import plot_rope_configuration
from link_bot_gazebo import gazebo_utils
from link_bot_planning import model_utils
from link_bot_planning.params import LocalEnvParams
from link_bot_planning.shooting_directed_control_sampler import ShootingDirectedControlSamplerInternal
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_pycommon import make_random_rope_configuration

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='gp')
    parser.add_argument("--n-samples", type=int, help='number of actions to sample', default=10)
    parser.add_argument("--n-examples", type=int, help='number of examples to run', default=10)
    parser.add_argument("--no-plot", action='store_true')

    args = parser.parse_args()

    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir, args.fwd_model_type)
    classifier_model = none_classifier.NoneClassifier()

    services = gazebo_utils.GazeboServices()

    # NOTE: these params don't matter at the moment, but they may soon
    local_env_params = LocalEnvParams(h_rows=100, w_cols=100, res=0.03)
    rope_length = fwd_model.hparams['dynamics_dataset_hparams']['rope_length']

    control_sampler = ShootingDirectedControlSamplerInternal(fwd_model=fwd_model,
                                                             classifier_model=classifier_model,
                                                             services=services,
                                                             local_env_params=local_env_params,
                                                             max_v=0.15,
                                                             n_samples=args.n_samples,
                                                             n_state=6,
                                                             n_local_env=100 * 100)

    n_no_progress = 0
    for i in range(args.n_examples):
        initial_state = make_random_rope_configuration(extent=[-2.5, 2.5, -2.5, 2.5],
                                                       n_state=fwd_model.n_state,
                                                       total_length=rope_length,
                                                       max_angle_rad=1)
        target_state = make_random_rope_configuration(extent=[-2.5, 2.5, -2.5, 2.5],
                                                      n_state=fwd_model.n_state,
                                                      total_length=rope_length,
                                                      max_angle_rad=1)

        reached_state, u, local_env, no_progress = control_sampler.sampleTo(np.expand_dims(initial_state, 0),
                                                                            np.expand_dims(target_state, 0))
        reached_state = np.squeeze(reached_state)
        u = np.squeeze(u)

        if no_progress:
            n_no_progress += 1

        if not args.no_plot:
            plt.figure()
            ax = plt.gca()
            plt.axis("equal")
            plot_rope_configuration(ax, initial_state, c='r', label='initial')
            plot_rope_configuration(ax, target_state, c='g', label='target')
            plot_rope_configuration(ax, reached_state, c='b', label='reached')
            plt.scatter(initial_state[4], initial_state[5], c='k')
            plt.scatter(target_state[4], target_state[5], c='k')
            plt.scatter(reached_state[4], reached_state[5], c='k')
            plt.quiver(initial_state[4], initial_state[5], u[0], u[1])
            plt.legend()
            plt.show()

    msg = "Found {} examples out of {} where we couldn't sample any controls that moved us towards the target configuration"
    print(msg.format(n_no_progress, args.n_examples))


if __name__ == '__main__':
    main()
