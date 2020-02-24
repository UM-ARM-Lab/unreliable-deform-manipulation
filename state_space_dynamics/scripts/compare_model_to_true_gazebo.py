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
from colorama import Fore
from matplotlib import animation

from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning import model_utils
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.ros_pycommon import make_trajectory_execution_request, get_occupancy_data, \
    trajectory_execution_response_to_numpy, get_start_states
from state_space_dynamics.base_forward_model import BaseForwardModel

tf.compat.v1.enable_eager_execution()


def visualize(root, predicted_points, actual_points, traj_idx):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    predicted_rope_handle, = ax.plot([], [], color='r', label='predicted')
    predicted_scatt = ax.scatter([], [], color='k', s=10)
    actual_rope_handle, = ax.plot([], [], color='b', label='actual')
    actual_scatt = ax.scatter([], [], color='k', s=10)
    ax.axis('equal')
    ax.set_xlim([-5.0, 5.0])
    ax.set_ylim([-5.0, 5.0])

    def update(t):
        predicted_xs = predicted_points[t, :, 0]
        predicted_ys = predicted_points[t, :, 1]
        predicted_rope_handle.set_data(predicted_xs, predicted_ys)
        predicted_scatt.set_offsets([predicted_xs[-1], predicted_ys[-1]])

        actual_xs = actual_points[t, :, 0]
        actual_ys = actual_points[t, :, 1]
        actual_rope_handle.set_data(actual_xs, actual_ys)
        actual_scatt.set_offsets([actual_xs[-1], actual_ys[-1]])

    anim = animation.FuncAnimation(fig, update, interval=250, frames=len(predicted_points))

    plt.legend()
    plt.tight_layout()

    outname = "compare_{}.gif".format(traj_idx)
    outname = root / outname
    anim.save(str(outname), writer='imagemagick', fps=4)

    plt.close()


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("model_dir", type=pathlib.Path, help='path to model')
    parser.add_argument("model_type", choices=['nn', 'llnn', 'rigid', 'obs'], default='obs', help='type of model')
    parser.add_argument("outdir", type=pathlib.Path, help="output metrics (and optionally visualizations) here")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-step-size', type=float, default=0.01)
    parser.add_argument('--n-trajs', type=int, default=100)
    parser.add_argument('--n-actions-per-traj', type=int, default=5)
    parser.add_argument('--p-kink', type=int, default=0.1)
    parser.add_argument('--kinked-actions', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rospy.init_node('compare_to_true_gazebo')

    fwd_model, _ = model_utils.load_generic_model(args.model_dir, args.model_type)

    # Start Services
    services = gazebo_utils.GazeboServices()

    # Setup for saving results
    root = args.outdir / "compare_to_gz_{}".format(int(time.time()))
    root.mkdir(exist_ok=True)
    metrics_filename = root / 'metrics.npz'
    print(Fore.CYAN + "Writing to {}".format(str(metrics_filename)) + Fore.RESET)

    all_metrics = {}
    for traj_idx in range(args.n_trajs):
        metrics = run_traj(args, services, fwd_model, traj_idx, root)
        # This basically assumes metrics is a flat dictionary of [str, np.ndarray]
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v.tolist())

        # Overwrite after every trajectory
        np.savez(metrics_filename, **all_metrics)


def run_traj(args,
             services: GazeboServices,
             fwd_model: BaseForwardModel,
             traj_idx: int,
             root: pathlib.Path):
    services.setup_env(args.verbose, real_time_rate=1.0, max_step_size=args.max_step_size, reset_world=True)
    services.pause(std_srvs.srv.EmptyRequest())

    if args.kinked_actions:
        actions = sample_kinked_action_sequence(args.n_actions_per_traj, args.p_kink)
    else:
        actions = sample_const_action_sequence(args.n_actions_per_traj)

    full_env_data = get_occupancy_data(env_w=fwd_model.full_env_params.w,
                                       env_h=fwd_model.full_env_params.h,
                                       res=fwd_model.full_env_params.res,
                                       services=services)
    state_keys = fwd_model.hparams['states_keys']
    print("get")
    start_states, link_bot_start_state, head_point = get_start_states(services, state_keys)

    print("pred")
    predicted_paths = fwd_model.predict(full_env=full_env_data.data,
                                         full_env_origin=full_env_data.origin,
                                         res=full_env_data.resolution[0],
                                         states=start_states,
                                         actions=actions)

    trajectory_execution_request = make_trajectory_execution_request(fwd_model.dt, actions)
    print("exec")
    traj_res = services.execute_trajectory(trajectory_execution_request)
    actual_paths = trajectory_execution_response_to_numpy(traj_res, None, services)
    print("done")

    error_metrics = {'actions': actions}
    for state_name, predicted_path in predicted_paths.items():
        actual_path = actual_paths[state_name]
        actual_path = actual_path.reshape([actual_path.shape[0], -1, 2])
        predicted_path = predicted_path.reshape([predicted_path.shape[0], -1, 2])
        errors = np.linalg.norm(predicted_path - actual_path, axis=2)

        if not args.no_plot:
            visualize(root, predicted_path, actual_path, traj_idx)
            print("mean error for state {}: {:5.3f}".format(state_name, np.mean(errors)))

        error_metrics[state_name] = errors

    return error_metrics


def sample_kinked_action_sequence(n_actions_per_traj, p_kink):
    actions = []
    a_t = np.random.uniform([-0.15, -0.15], [0.15, 0.15])
    for t in range(n_actions_per_traj):
        r = np.random.uniform(0.0, 1.0)
        if r < p_kink:
            a_t = np.random.uniform([-0.15, -0.15], [0.15, 0.15])
        actions.append(a_t)
    actions = np.array(actions)
    return actions


def sample_const_action_sequence(n_actions_per_traj):
    actions = []
    theta = np.random.uniform(-np.pi, np.pi)
    for t in range(n_actions_per_traj):
        a_t = np.array([np.cos(theta) * 0.15, np.sin(theta) * 0.15])
        actions.append(a_t)
    actions = np.array(actions)
    return actions


if __name__ == '__main__':
    main()
