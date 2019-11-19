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
from link_bot_gazebo.srv import LinkBotStateRequest
from link_bot_planning import model_utils

tf.enable_eager_execution()


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
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=pathlib.Path, help='path to model')
    parser.add_argument("model_type", choices=['llnn', 'gp', 'rigid'], default='llnn', help='type of model')
    parser.add_argument("outdir", type=pathlib.Path, help="output metrics (and optionally visualizations) here")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n-trajs', type=int, default=100)
    parser.add_argument('--n-actions-per-traj', type=int, default=100)
    parser.add_argument('--n_parallel_calls-kink', type=int, default=0.1)
    parser.add_argument('--no-plot', action='store_true')

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


def run_traj(args, services, fwd_model, traj_idx, root):
    services.reset_gazebo_environment(reset_model_poses=True)
    services.pause(std_srvs.srv.EmptyRequest())
    state_req = LinkBotStateRequest()
    state = services.get_state.call(state_req)
    initial_rope_configuration = np.array([[p.x, p.y] for p in state.points]).flatten()

    actions = sample_kinked_action_sequence(args.n_actions_per_traj, args.p_kink)

    predicted_points = fwd_model.predict(np.expand_dims(initial_rope_configuration, axis=0), np.expand_dims(actions, axis=0))
    predicted_points = predicted_points[0]
    trajectory_execution_request = gazebo_utils.make_trajectory_execution_request(fwd_model.dt, actions)
    traj_res = services.execute_trajectory(trajectory_execution_request)
    actual_points, _ = gazebo_utils.trajectory_execution_response_to_numpy(traj_res,
                                                                           None,
                                                                           services)
    actual_points = actual_points.reshape([actual_points.shape[0], 3, 2])
    position_errors = np.linalg.norm(predicted_points - actual_points, axis=2)
    if not args.no_plot:
        visualize(root, predicted_points, actual_points, traj_idx)
        print("mean error: {:5.3f}".format(np.mean(position_errors)))

    return {'error': position_errors}


def sample_kinked_action_sequence(n_actions_per_traj, p_kink):
    actions = []
    a_t = np.random.uniform([-0.15, -0.15], [0.15, 0.15])
    for t in range(n_actions_per_traj):
        r = np.random.uniform(0.0, 1.0)
        if r < p_kink:
            a_t = np.random.uniform([-0.15, -0.15], [0.15, 0.15])
        a_t_noisy = a_t + np.random.multivariate_normal([0, 0], np.eye(2) * 1e-5)
        actions.append(a_t_noisy)
    actions = np.array(actions)
    return actions


if __name__ == '__main__':
    main()
