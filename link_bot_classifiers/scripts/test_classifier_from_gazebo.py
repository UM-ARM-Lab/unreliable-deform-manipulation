#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import rospy
import tensorflow as tf

from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning import model_utils, classifier_utils
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_pycommon import flatten_points
from link_bot_pycommon.ros_pycommon import get_local_occupancy_data, get_occupancy_data, get_states_dict
from peter_msgs.srv import LinkBotStateRequest

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


def make_float32(d):
    for k, s_k in d.items():
        d[k] = s_k.astype(np.float32)
    return d


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("classifier_model_dir", help="classifier", type=pathlib.Path)
    parser.add_argument('--no-plot', action='store_true', help="don't show plots, useful for debugging")
    parser.add_argument('-v', type=float, default=0.3, help='speed of test actions')

    args = parser.parse_args()

    # use forward model to predict given the input action
    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir)
    classifier_model = classifier_utils.load_generic_model(args.classifier_model_dir, fwd_model.scenario)
    full_env_params = fwd_model.full_env_params

    rospy.init_node('test_classifier_from_gazebo')

    service_provider = GazeboServices()

    full_env_data = get_occupancy_data(env_w_m=full_env_params.w,
                                       env_h_m=full_env_params.h,
                                       res=full_env_params.res,
                                       service_provider=service_provider,
                                       robot_name=fwd_model.scenario.robot_name())

    state = get_states_dict(service_provider, ['link_bot'])

    v = args.v
    test_inputs = [
        (v, 135),
        (v, 90),
        (v, 45),
        (v, 180),
        (0, 0),
        (v, 0),
        (v, 225),
        (v, 270),
        (v, 315),
    ]

    fig, axes = plt.subplots(3, 3)
    for k, (v, theta_deg) in enumerate(test_inputs):
        theta_rad = np.deg2rad(theta_deg)
        vx = np.cos(theta_rad) * v
        vy = np.sin(theta_rad) * v
        actions = np.array([[vx, vy]])

        pred_states = fwd_model.propagate(full_env=full_env_data.data,
                                          full_env_origin=full_env_data.origin,
                                          res=full_env_data.resolution,
                                          start_states=state,
                                          actions=actions)
        next_state = pred_states[1]

        state = make_float32(state)
        next_state = make_float32(next_state)
        states_sequence = [state, next_state]

        accept_probability = classifier_model.check_constraint(full_env=full_env_data.data,
                                                               full_env_origin=full_env_data.origin,
                                                               res=full_env_data.resolution,
                                                               states_sequence=states_sequence,
                                                               actions=actions)
        accept_probability = float(accept_probability)
        prediction = 1 if accept_probability > 0.5 else 0
        title = 'P(accept) = {:04.3f}%'.format(100 * accept_probability)

        print("v={:04.3f}m/s theta={:04.3f}deg    p(accept)={:04.3f}".format(v, theta_deg, accept_probability))
        if not args.no_plot:
            i, j = np.unravel_index(k, [3, 3])
            plot_classifier_data(ax=axes[i, j],
                                 actual_env=full_env_data.data,
                                 actual_env_extent=full_env_data.extent,
                                 planned_state=state['link_bot'],
                                 planned_next_state=next_state['link_bot'],
                                 title=title,
                                 label=prediction)

    if not args.no_plot:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
