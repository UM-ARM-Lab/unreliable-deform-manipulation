#!/usr/bin/env python

import argparse
import json
import pathlib
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import rospy
import std_srvs
import tensorflow as tf
from matplotlib import animation

from link_bot_gazebo import gazebo_services
from link_bot_planning import model_utils, classifier_utils, ompl_viz
from link_bot_planning.get_scenario import get_scenario
from link_bot_planning.plan_and_execute import execute_plan
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_sdf_utils import OccupancyData
from link_bot_pycommon.ros_pycommon import make_trajectory_execution_request, trajectory_execution_response_to_numpy, \
    get_occupancy_data, get_states_dict
from victor import victor_services

tf.compat.v1.enable_eager_execution()


def visualize(args, env_data: OccupancyData, predicted_paths, actual_paths, p_accept_s):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    predicted_rope_handles = {}
    predicted_scatts = {}
    actual_rope_handles = {}
    actual_scatts = {}
    for state_name in predicted_paths.keys():
        predicted_rope_handle, = ax.plot([], [], color='r', label='predicted {}'.format(state_name))
        predicted_rope_handles[state_name] = predicted_rope_handle
        predicted_scatt = ax.scatter([], [], color='k', s=10)
        predicted_scatts[state_name] = predicted_scatt
        actual_rope_handle, = ax.plot([], [], color='b', label='actual {}'.format(state_name))
        actual_rope_handles[state_name] = actual_rope_handle
        actual_scatt = ax.scatter([], [], color='k', s=10)
        actual_scatts[state_name] = actual_scatt

    ax.axis('equal')
    ax.set_xlim([env_data.extent[0], env_data.extent[1]])
    ax.set_ylim([env_data.extent[2], env_data.extent[3]])
    ax.imshow(np.flipud(env_data.data), extent=env_data.extent)

    def update(t):
        for state_name, predicted_path in predicted_paths.items():
            actual_path = actual_paths[state_name]
            predicted_xs = predicted_path[t, :, 0]
            predicted_ys = predicted_path[t, :, 1]
            predicted_rope_handles[state_name].set_data(predicted_xs, predicted_ys)
            predicted_scatts[state_name].set_offsets([predicted_xs[-1], predicted_ys[-1]])

            actual_xs = actual_path[t, :, 0]
            actual_ys = actual_path[t, :, 1]
            actual_rope_handles[state_name].set_data(actual_xs, actual_ys)
            actual_scatts[state_name].set_offsets([actual_xs[-1], actual_ys[-1]])

    T = predicted_paths['link_bot'].shape[0]
    anim = animation.FuncAnimation(fig, update, interval=250, frames=T)

    plt.legend()
    plt.tight_layout()

    if args.outdir is not None:
        outname = "model_vs_true_{}.gif".format(int(time.time()))
        outname = args.outdir / outname
        anim.save(str(outname), writer='imagemagick', fps=4)

    plt.show()


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("env_type", choices=['victor', 'gazebo'], default='gazebo', help='victor or gazebo')
    parser.add_argument("params", type=pathlib.Path, help='json file')
    parser.add_argument("actions", type=pathlib.Path, help='csv file of actions')
    parser.add_argument("--outdir", type=pathlib.Path, help="output visualizations here")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--real-time-rate', type=float, default=10.0)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    plt.style.use("paper")

    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rospy.init_node('compare_to_true_given_actions')

    # Start Services
    if args.env_type == 'victor':
        service_provider = victor_services.VictorServices()
    else:
        service_provider = gazebo_services.GazeboServices()

    params = json.load(args.params.open('r'))

    scenario = get_scenario(params['scenario'])
    fwd_model, _ = model_utils.load_generic_model(params['fwd_model_dir'])
    classifier_model = classifier_utils.load_generic_model(params['classifier_model_dir'], fwd_model.scenario)

    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=args.real_time_rate,
                               reset_robot=params['reset_robot'],
                               max_step_size=fwd_model.max_step_size)
    service_provider.pause(std_srvs.srv.EmptyRequest())

    full_env_data = get_occupancy_data(env_w_m=classifier_model.full_env_params.w,
                                       env_h_m=classifier_model.full_env_params.h,
                                       res=fwd_model.full_env_params.res,
                                       service_provider=service_provider,
                                       robot_name=scenario.robot_name())

    start_states = get_states_dict(service_provider, fwd_model.states_keys)

    actions = np.genfromtxt(args.actions, delimiter=',')
    T = actions.shape[0]
    actions = actions.reshape([-1, fwd_model.n_action])

    predicted_path = fwd_model.propagate(full_env=full_env_data.data,
                                         full_env_origin=full_env_data.origin,
                                         res=full_env_data.resolution,
                                         start_states=start_states,
                                         actions=actions)
    # Check classifier
    accept_probabilities = []
    for t in range(1, T):
        accept_probability = classifier_model.check_constraint(full_env=full_env_data.data,
                                                               full_env_origin=full_env_data.origin,
                                                               res=full_env_data.resolution,
                                                               states_sequence=predicted_path[:t + 1],
                                                               actions=actions)
        accept_probabilities.append(accept_probability)

    actual_path = execute_plan(service_provider, fwd_model.dt, actions)

    anim = ompl_viz.plan_vs_execution(full_env_data.data,
                                      full_env_data.extent,
                                      scenario,
                                      goal=None,
                                      accept_probabilities=accept_probabilities,
                                      planned_path=predicted_path,
                                      actual_path=actual_path)
    if args.outdir:
        outfilename = args.outdir / 'compare_model_to_gazebo_with_classifier' / 'actions_{}.gif'.format(args.actions.stem)
        anim.save(outfilename, writer='imagemagick', dpi=100)
    plt.show()


if __name__ == '__main__':
    main()
