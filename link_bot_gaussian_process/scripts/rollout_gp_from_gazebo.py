#!/usr/bin/env python

import std_srvs
import argparse
import os
from matplotlib import animation
import random

import matplotlib.pyplot as plt
import numpy as np
import rospy
import tensorflow as tf

from link_bot_data.visualization import plottable_rope_configuration
from link_bot_gaussian_process import link_bot_gp
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.srv import LinkBotStateRequest


def visualize(predicted_traj):
    fig, axes = plt.subplots()

    rope_handle, = axes.plot([], [], color='r')
    head_scatt = axes.scatter([], [], color='k', s=100)
    other_points_scatt = axes.scatter([], [], color='k', s=50)
    axes.set_xlim([-1.0, 1.0])
    axes.set_ylim([-1.0, 1.0])

    def update(t):
        rope_config = predicted_traj[t][0]
        xs, ys = plottable_rope_configuration(rope_config)
        rope_handle.set_data(xs, ys)
        head_scatt.set_offsets([xs[-1], ys[-1]])
        other_points_scatt.set_offsets(np.stack([xs[:-1], ys[:-1]], axis=1))

    _ = animation.FuncAnimation(fig, update, interval=250, frames=len(predicted_traj))
    plt.show()


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("gp_model_dir")
    parser.add_argument("actions")
    parser.add_argument("--outdir", help="output visualizations here")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    rospy.init_node('gazebo_compare_rollout')

    # Start Services
    services = gazebo_utils.GazeboServices()

    services.reset_gazebo_environment(reset_model_poses=False)
    services.pause(std_srvs.srv.EmptyRequest())

    state_req = LinkBotStateRequest()
    state = services.get_state.call(state_req)
    initial_rope_configuration = np.array([[p.x, p.y] for p in state.points]).flatten()

    actions = np.genfromtxt(args.actions, delimiter=',')

    fwd_gp_model = link_bot_gp.LinkBotGP()
    fwd_gp_model.load(os.path.join(args.gp_model_dir, 'fwd_model'))

    s = np.expand_dims(initial_rope_configuration, axis=0)
    predicted_traj = [s]
    for action in actions:
        s_next = fwd_gp_model.fwd_act(s, np.expand_dims(action, axis=0))
        predicted_traj.append(s_next)
        s = s_next

    visualize(predicted_traj)


if __name__ == '__main__':
    main()
