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
    fig, axes = plt.subplots(nrows=1, ncols=2)

    points = np.array(predicted_traj).reshape([-1, 3, 2])
    head_to_mid_lengths = np.linalg.norm(points[:, 1] - points[:, 0], axis=1)
    mid_to_tail_lengths = np.linalg.norm(points[:, 2] - points[:, 1], axis=1)
    axes[1].plot(head_to_mid_lengths, label='head to mid dist')
    axes[1].plot(mid_to_tail_lengths, label='mid to tail dist')

    rope_handle, = axes[0].plot([], [], color='r')
    head_scatt = axes[0].scatter([], [], color='k', s=100)
    other_points_scatt = axes[0].scatter([], [], color='k', s=50)
    axes[0].set_xlim([-5.0, 5.0])
    axes[0].set_ylim([-5.0, 5.0])

    def update(t):
        rope_config = predicted_traj[t][0]
        xs, ys = plottable_rope_configuration(rope_config)
        rope_handle.set_data(xs, ys)
        head_scatt.set_offsets([xs[-1], ys[-1]])
        other_points_scatt.set_offsets(np.stack([xs[:-1], ys[:-1]], axis=1))

    anim = animation.FuncAnimation(fig, update, interval=250, frames=len(predicted_traj))
    plt.legend()
    anim.save('gp_rollout.gif', writer='imagemagick')
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
