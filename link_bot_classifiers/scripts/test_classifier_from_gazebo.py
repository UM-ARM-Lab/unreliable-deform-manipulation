#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import rospy
import tensorflow as tf

from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import get_local_occupancy_data, GazeboServices
from link_bot_gazebo.srv import LinkBotStateRequest
from link_bot_planning import model_utils, classifier_utils
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_pycommon.args import my_formatter

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='llnn')
    parser.add_argument("classifier_model_dir", help="classifier", type=pathlib.Path)
    parser.add_argument("classifier_model_type", choices=['none', 'raster'], default='raster')
    parser.add_argument('v', type=float, help='speed in m/s')
    parser.add_argument('theta', type=float, help='direction of velocity in DEGREES relative to +x axis (right, east)')
    parser.add_argument('--rows', type=int, help='number of rows in local environment', default=100)
    parser.add_argument('--cols', type=int, help='number of cols in local environment', default=100)
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument('--no-plot', action='store_true', help="don't show plots, useful for debugging")

    args = parser.parse_args()

    rospy.init_node('test_classifier_from_gazebo')

    services = GazeboServices()

    state_req = LinkBotStateRequest()
    link_bot_state = services.get_state(state_req)
    head_idx = link_bot_state.link_names.index("head")
    head_point = link_bot_state.points[head_idx]
    head_point = np.array([head_point.x, head_point.y])
    local_env_data = get_local_occupancy_data(args.rows, args.cols, args.res, center_point=head_point, services=services)

    # use forward model to predict given the input action
    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir, args.fwd_model_type)
    classifier_model = classifier_utils.load_generic_model(args.classifier_model_dir, args.classifier_model_type)

    state = np.expand_dims(gazebo_utils.points_to_config(link_bot_state.points), axis=0)
    theta_rad = np.deg2rad(args.theta)
    vx = np.cos(theta_rad) * args.v
    vy = np.sin(theta_rad) * args.v
    action = np.array([[[vx, vy]]])
    next_state = fwd_model.predict(state, action)
    next_state = np.reshape(next_state, [2, 1, 6])[1]

    accept_probability = classifier_model.predict(local_env_data, state, next_state)
    prediction = 1 if accept_probability > 0.5 else 0
    title = 'P(accept) = {:04.3f}%'.format(100 * accept_probability)

    if not args.no_plot:
        # conv_filters = classifier_model.net.conv_layers[0].get_weights()[0]
        # fig, axes = plt.subplots(5, 9)
        # for ax in axes.flatten():
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.axis("off")
        # for channel_idx, filter_idx in np.ndindex(7, 6):
        #     n = channel_idx * 6 + filter_idx
        #     r, c = np.unravel_index(n, [5, 9])
        #     ax = axes[r, c]
        #     filter = conv_filters[channel_idx, filter_idx]
        #     ax.imshow(np.flipud(filter), vmin=-1, vmax=1)

        plot_classifier_data(planned_env=local_env_data.data,
                             planned_env_extent=local_env_data.extent,
                             planned_state=state[0],
                             planned_next_state=next_state[0],
                             state=None,
                             next_state=None,
                             title=title,
                             actual_env=None,
                             actual_env_extent=None,
                             label=prediction)
        plt.show()
    else:
        print(title)


if __name__ == '__main__':
    main()
