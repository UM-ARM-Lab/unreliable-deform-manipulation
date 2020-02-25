#!/usr/bin/env python

import argparse
import pathlib
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import rospy
import std_srvs
import tensorflow as tf
from matplotlib import animation

from link_bot_gazebo import gazebo_utils
from link_bot_planning import model_utils
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_sdf_utils import OccupancyData
from link_bot_pycommon.ros_pycommon import make_trajectory_execution_request, trajectory_execution_response_to_numpy, \
    get_occupancy_data, get_start_states
from victor import victor_utils

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
    ax.set_xlim([-5.0, 5.0])
    ax.set_ylim([-5.0, 5.0])
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
    parser.add_argument("model_dir", type=pathlib.Path, help='path to model')
    parser.add_argument("model_type", choices=['nn', 'obs', 'rigid'], default='nn', help='type of model')
    parser.add_argument("classifier_dir", type=pathlib.Path, help='path to model')
    parser.add_argument("classifier_type", choices=['raster', 'collision', 'none'], default='raster', help='type of classifier')
    parser.add_argument("actions", type=pathlib.Path, help='csv file of actions')
    parser.add_argument("--outdir", type=pathlib.Path, help="output visualizations here")
    parser.add_argument('--max-step-size', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rospy.init_node('compare_to_true_given_actions')

    # Start Services
    if args.env_type == 'victor':
        services = victor_utils.VictorServices()
    else:
        services = gazebo_utils.GazeboServices()

    services.setup_env(verbose=args.verbose,
                       real_time_rate=10.0,
                       reset_gripper_to=None,
                       max_step_size=args.max_step_size)
    services.pause(std_srvs.srv.EmptyRequest())

    fwd_model, _ = model_utils.load_generic_model(args.model_dir, args.model_type)

    full_env_data = get_occupancy_data(env_w=fwd_model.full_env_params.w,
                                       env_h=fwd_model.full_env_params.h,
                                       res=fwd_model.full_env_params.res,
                                       services=services)
    state_keys = fwd_model.hparams['states_keys']
    start_states, link_bot_start_state, head_point = get_start_states(services, state_keys)

    actions = np.genfromtxt(args.actions, delimiter=',')
    actions = actions.reshape([-1, fwd_model.n_control])

    predicted_paths = fwd_model.predict(full_env=full_env_data.data,
                                        full_env_origin=full_env_data.origin,
                                        res=full_env_data.resolution[0],
                                        states=start_states,
                                        actions=actions)

    trajectory_execution_request = make_trajectory_execution_request(fwd_model.dt, actions)
    print(trajectory_execution_request)
    traj_res = services.execute_trajectory(trajectory_execution_request)
    actual_paths = trajectory_execution_response_to_numpy(traj_res, None, services)

    # Reshape into points for drawing
    for state_name, predicted_path in predicted_paths.items():
        actual_path = actual_paths[state_name]
        actual_paths[state_name] = actual_path.reshape([actual_path.shape[0], -1, 2])
        predicted_paths[state_name] = predicted_path.reshape([predicted_path.shape[0], -1, 2])

    # Compute some error metrics
    for state_name, predicted_path in predicted_paths.items():
        actual_path = actual_paths[state_name]
        errors = np.linalg.norm(predicted_path - actual_path, axis=2)
        print("mean error for state {}: {:5.3f}".format(state_name, np.mean(errors)))

    # animate prediction versus actual
    visualize(args, full_env_data, predicted_paths, actual_paths, 0)


if __name__ == '__main__':
    main()
