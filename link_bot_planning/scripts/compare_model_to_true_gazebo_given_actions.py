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
from link_bot_gazebo.gazebo_utils import get_local_occupancy_data
from link_bot_gazebo.srv import LinkBotStateRequest, WorldControlRequest
from link_bot_planning import model_utils, classifier_utils

tf.enable_eager_execution()


def visualize(args, env_data, predicted_points, actual_points, p_accept_s):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    predicted_rope_handle, = ax.plot([], [], color='r', label='predicted')
    predicted_scatt = ax.scatter([], [], color='k', s=10)
    actual_rope_handle, = ax.plot([], [], color='b', label='actual')
    actual_scatt = ax.scatter([], [], color='k', s=10)
    ax.axis('equal')
    ax.set_xlim([-5.0, 5.0])
    ax.set_ylim([-5.0, 5.0])
    ax.imshow(env_data.image, extent=env_data.extent)

    def update(t):
        predicted_xs = predicted_points[t, :, 0]
        predicted_ys = predicted_points[t, :, 1]
        predicted_rope_handle.set_data(predicted_xs, predicted_ys)
        if t < len(p_accept_s):
            if p_accept_s[t] > 0.5:
                predicted_rope_handle.set_color('w')
            else:
                predicted_rope_handle.set_color('k')
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
    parser.add_argument("model_type", choices=['llnn', 'gp', 'rigid', 'nn'], default='llnn', help='type of model')
    parser.add_argument("classifier_dir", type=pathlib.Path, help='path to model')
    parser.add_argument("classifier_type", choices=['raster', 'collision', 'none'], default='raster', help='type of classifier')
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
    services.reset_gazebo_environment(reset_model_poses=True)
    services.pause(std_srvs.srv.EmptyRequest())

    step = WorldControlRequest()
    step.steps = 5000
    services.world_control(step)  # this will block until stepping is complete

    state_req = LinkBotStateRequest()
    state = services.get_state.call(state_req)
    initial_rope_configuration = np.array([[p.x, p.y] for p in state.points]).flatten()

    actions = np.genfromtxt(args.actions, delimiter=',')
    actions = np.atleast_2d(actions)

    fwd_model, _ = model_utils.load_generic_model(args.model_dir, args.model_type)
    classifier_model = classifier_utils.load_generic_model(args.classifier_dir, args.classifier_type)
    dt = fwd_model.dt

    env_data = get_local_occupancy_data(rows=200,
                                        cols=200,
                                        res=fwd_model.local_env_params.res,
                                        center_point=np.array([0, 0]),
                                        services=services)
    predicted_points = fwd_model.predict(local_env_data=[env_data],
                                         first_states=np.expand_dims(initial_rope_configuration, axis=0),
                                         actions=np.expand_dims(actions, axis=0))
    predicted_points = predicted_points[0]

    p_accept_s = []
    for i in range(actions.shape[0] - 1):
        center_point = np.array([predicted_points[i][2, 0], predicted_points[i][2, 1]])
        local_env_data = get_local_occupancy_data(rows=fwd_model.local_env_params.h_rows,
                                                  cols=fwd_model.local_env_params.w_cols,
                                                  res=fwd_model.local_env_params.res,
                                                  center_point=center_point,
                                                  services=services)
        p_accept = classifier_model.predict(local_env_data_s=[local_env_data],
                                            s1_s=predicted_points[i].reshape([1, 6]),
                                            s2_s=predicted_points[i + 1].reshape([1, 6]))
        print(p_accept)
        p_accept_s.append(p_accept)

    trajectory_execution_request = gazebo_utils.make_trajectory_execution_request(dt, actions)
    traj_res = services.execute_trajectory(trajectory_execution_request)

    actual_points, _ = gazebo_utils.trajectory_execution_response_to_numpy(traj_res,
                                                                           None,
                                                                           services)
    actual_points = actual_points.reshape([actual_points.shape[0], 3, 2])

    position_errors = np.linalg.norm(predicted_points - actual_points, axis=2)
    print("mean error: {:5.3f}".format(np.mean(position_errors)))

    if not args.no_plot:
        visualize(args, env_data, predicted_points, actual_points, p_accept_s)


if __name__ == '__main__':
    main()
