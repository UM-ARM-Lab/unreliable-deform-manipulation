#!/usr/bin/env python

import argparse
import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import rospy
import tensorflow as tf

from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import get_local_occupancy_data, GazeboServices, get_sdf_data
from link_bot_gazebo.srv import LinkBotStateRequest
from link_bot_planning import model_utils, classifier_utils
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_pycommon import link_bot_pycommon
from link_bot_pycommon.args import my_formatter

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


def main():
    # get the environment from gazebo
    # sample random configurations in the environment, and a random action. Propogate through the model
    # make predictions through each model in the ensemble
    # plot their sigmoid values

    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['nn', 'gp', 'llnn', 'rigid'], default='nn')
    parser.add_argument("ensemble_directory", help="classifier", type=pathlib.Path)
    parser.add_argument("classifier_model_type", choices=['none', 'raster'], default='raster')
    parser.add_argument("--n-configs", '-n', help="number of random configs to test", type=int, default=10)
    parser.add_argument("--velocity", help="velocity of action to sample", type=float, default=0.15)
    parser.add_argument('--no-plot', action='store_true', help="don't show plots, useful for debugging")
    parser.add_argument('--outdir', type=pathlib.Path, help="save results here")

    args = parser.parse_args()

    rospy.init_node('visualize_ensemble_predictions')

    services = GazeboServices()

    # Load the forward model and the classifiers
    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir, args.fwd_model_type)
    ensemble = {}
    for classifier_dir in args.ensemble_directory.iterdir():
        classifier_model = classifier_utils.load_generic_model(classifier_dir, args.classifier_model_type)
        ensemble[classifier_dir] = classifier_model

    rope_length = fwd_model.hparams['dynamics_dataset_hparams']['rope_length']
    full_sdf_data = get_sdf_data(env_h=5, env_w=5, res=fwd_model.local_env_params.res, services=services)

    # TODO: batch predictions
    for i in range(args.n_configs):
        # pick random configuration
        state = link_bot_pycommon.make_random_rope_configuration(full_sdf_data.extent, fwd_model.n_state, rope_length,
                                                                 max_angle_rad=1)

        # get local environment
        head_point = np.array([state[4], state[5]])
        local_env_data = get_local_occupancy_data(rows=fwd_model.local_env_params.h_rows,
                                                  cols=fwd_model.local_env_params.w_cols,
                                                  res=fwd_model.local_env_params.res,
                                                  center_point=head_point,
                                                  services=services)

        # pick random action
        theta_rad = np.random.uniform(-np.pi, np.pi)
        vx = np.cos(theta_rad) * args.velocity
        vy = np.sin(theta_rad) * args.velocity
        action = np.array([[vx, vy]])

        # forward simulate
        next_state_s = fwd_model.predict(local_env_data=[local_env_data],
                                         state=np.expand_dims(state, axis=0),
                                         actions=np.expand_dims(action, axis=0))
        next_state = np.reshape(next_state_s, [2, 1, -1])[1, 0]

        predictions = []
        for classifier_model in ensemble.values():
            accept_probabilities = classifier_model.predict(local_env_data=[local_env_data],
                                                            s1_s=np.expand_dims(state, axis=0),
                                                            s2_s=np.expand_dims(next_state, axis=0))
            accept_probability = accept_probabilities[0]
            predictions.append(accept_probability)

        if not args.no_plot:
            fig, axes = plt.subplots(1, 2)
            axes[1].hist(predictions, bins=np.linspace(0, 1, 10))
            axes[1].set_xlabel("probability of accepting")
            axes[1].set_ylabel("count")

            axes[0].imshow(full_sdf_data.image < 0, extent=full_sdf_data.extent)
            x0 = local_env_data.extent[0]
            x1 = local_env_data.extent[1]
            y0 = local_env_data.extent[2]
            y1 = local_env_data.extent[3]
            axes[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color='w')
            plot_classifier_data(planned_env=local_env_data.data,
                                 planned_env_extent=local_env_data.extent,
                                 planned_env_origin=local_env_data.origin,
                                 res=local_env_data.resolution,
                                 planned_state=state,
                                 action=action[0],
                                 planned_next_state=next_state,
                                 state=None,
                                 next_state=None,
                                 title='',
                                 actual_env=None,
                                 actual_env_extent=None,
                                 label=None,
                                 ax=axes[0])
            if args.outdir:
                root = args.outdir / 'viz_{}'.format(int(time.time()))
                root.mkdir()
                data_file = root / 'viz_example_data_{}.json'.format(i)
                data = {
                    'state': state.tolist(),
                    'action': action.tolist(),
                    'planned_state': next_state.tolist(),
                    'predictions': predictions,
                    'local_env/env': local_env_data.data.tolist(),
                    'local_env/extent': local_env_data.extent,
                    'local_env/origin': local_env_data.origin.tolist(),
                }
                json.dump(data, data_file.open('w'))
                image_file = root / 'viz_example_{}.png'.format(i)
                plt.savefig(image_file)
            plt.show()


if __name__ == '__main__':
    main()
