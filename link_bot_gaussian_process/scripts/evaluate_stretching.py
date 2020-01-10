#!/usr/bin/env python

import argparse
import pathlib
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import animation

from link_bot_data.visualization import plottable_rope_configuration
from link_bot_gaussian_process import link_bot_gp
from link_bot_pycommon.link_bot_pycommon import make_random_rope_configuration


def visualize(outdir, predicted_traj, action, mid_to_tail_lengths, head_to_mid_lengths):
    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[1].plot(head_to_mid_lengths, label='head to mid dist')
    axes[1].plot(mid_to_tail_lengths, label='mid to tail dist')

    rope_handle, = axes[0].plot([], [], color='r')
    head_scatt = axes[0].scatter([], [], color='k', s=10)
    other_points_scatt = axes[0].scatter([], [], color='k', s=5)
    axes[0].set_xlim([-5.0, 5.0])
    axes[0].set_ylim([-5.0, 5.0])
    axes[1].set_ylim([0, 0.5])

    def update(t):
        rope_config = predicted_traj[t][0]
        xs, ys = plottable_rope_configuration(rope_config)
        rope_handle.set_data(xs, ys)
        head_scatt.set_offsets([xs[-1], ys[-1]])
        other_points_scatt.set_offsets(np.stack([xs[:-1], ys[:-1]], axis=1))

    anim = animation.FuncAnimation(fig, update, interval=250, frames=len(predicted_traj))
    plt.legend()
    plt.tight_layout()
    axes[0].set_title("Rollout (action={})".format(np.array2string(action)))
    axes[1].set_title("distance")

    if outdir:
        outname = outdir / "eval_stretching_rollout_{}.gif".format(int(time.time()))
        anim.save(outname, writer='imagemagick', fps=30)

    plt.show()


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("gp_model_dir", type=pathlib.Path)
    parser.add_argument("--outdir", help="output visualizations here", type=pathlib.Path)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--n-examples', type=int, default=10)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    fwd_gp_model = link_bot_gp.LinkBotGP()
    fwd_gp_model.load(args.gp_model_dir / 'fwd_model')

    final_deltas = []
    for i in range(args.n_examples):
        config = make_random_rope_configuration([-5, 5, -5, 5], total_length=0.23)
        action = np.random.uniform(-.15, .15, size=[2])
        actions = np.tile(action, [10, 1])
        s = np.expand_dims(config, axis=0)
        predicted_traj = [s]
        for action in actions:
            s_next = fwd_gp_model.fwd_act(s, np.expand_dims(action, axis=0))
            predicted_traj.append(s_next)
            s = s_next

        points = np.array(predicted_traj).reshape([-1, 3, 2])
        head_to_mid_lengths = np.linalg.norm(points[:, 2] - points[:, 1], axis=1)
        mid_to_tail_lengths = np.linalg.norm(points[:, 1] - points[:, 0], axis=1)

        nominal_length = 0.23
        final_delta = np.abs(head_to_mid_lengths[-1] - nominal_length) + np.abs(mid_to_tail_lengths[-1] - nominal_length)
        final_deltas.append(final_delta)

        if not args.no_plot:
            visualize(args.outdir, predicted_traj, action, mid_to_tail_lengths, head_to_mid_lengths)

    mean_final_delta = np.mean(final_deltas)
    print("mean change in total rope length {:0.4f}m".format(mean_final_delta))


if __name__ == '__main__':
    main()
