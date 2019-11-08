#!/usr/bin/env python

import argparse
import pathlib
import random
import time

import matplotlib.pyplot as plt
import std_srvs
import numpy as np
import rospy
import tensorflow as tf
from matplotlib import animation

from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_gazebo.srv import LinkBotStateRequest, LinkBotTrajectoryRequest
from link_bot_planning import model_utils
from link_bot_planning.params import LocalEnvParams

tf.enable_eager_execution()


def visualize(args, predicted_points, actual_points):
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

    if args.outdir is not None:
        outname = "llnn_vs_true_{}.gif".format(int(time.time()))
        outname = args.outdir / outname
        anim.save(str(outname), writer='imagemagick', fps=4)

    plt.show()


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=pathlib.Path, help='path to model')
    parser.add_argument("model_type", choices=['llnn', 'gp', 'rigid'], default='llnn', help='type of model')
    parser.add_argument("actions", type=pathlib.Path, help='csv file of actions')
    parser.add_argument("--outdir", type=pathlib.Path, help="output visualizations here")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-plot', action='store_true')

    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rospy.init_node('compare_to_true_gazebo_given_actions')

    # Start Services
    services = gazebo_utils.GazeboServices()
    services.reset_gazebo_environment(reset_model_poses=False)
    services.pause(std_srvs.srv.EmptyRequest())

    state_req = LinkBotStateRequest()
    state = services.get_state.call(state_req)
    initial_rope_configuration = np.array([[p.x, p.y] for p in state.points]).flatten()

    actions = np.genfromtxt(args.actions, delimiter=',')
    actions = np.atleast_2d(actions)

    fwd_model, _ = model_utils.load_generic_model(args.model_dir, args.model_type)
    dt = fwd_model.dt

    predicted_points = fwd_model.predict(np.expand_dims(initial_rope_configuration, axis=0), np.expand_dims(actions, axis=0))
    predicted_points = predicted_points[0]
    # predicted_points = predicted_points.reshape([predicted_points.shape[0], -1])

    trajectory_execution_request = gazebo_utils.make_trajectory_execution_request(dt, actions)
    traj_res = services.execute_trajectory(trajectory_execution_request)

    actual_points, _ = gazebo_utils.trajectory_execution_response_to_numpy(traj_res,
                                                                        None,
                                                                        services)
    actual_points = actual_points.reshape([actual_points.shape[0], 3, 2])

    position_errors = np.linalg.norm(predicted_points - actual_points, axis=2)
    print("mean error: {:5.3f}".format(np.mean(position_errors)))

    if not args.no_plot:
        visualize(args, predicted_points, actual_points)


if __name__ == '__main__':
    main()
