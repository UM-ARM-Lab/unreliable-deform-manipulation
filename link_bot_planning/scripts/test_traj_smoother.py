#!/usr/bin/env python
import argparse
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pathlib

import numpy as np
import tensorflow as tf

from link_bot_planning import model_utils
from link_bot_planning.trajectory_smoother import TrajectorySmoother
from link_bot_pycommon.args import my_formatter

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


def mock_fwd_model(state, actions):
    predictions = [state]
    for t in range(actions.shape[0]):
        action = actions[t]
        points = tf.reshape(state, [3, 2])
        next_points = points + action
        next_state = tf.reshape(next_points, [1, 6])
        predictions.append(next_state)
        state = next_state
    return predictions


def main():
    plt.style.use("slides")

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('fwd_model_dir', type=pathlib.Path)
    parser.add_argument('fwd_model_type', type=str)
    parser.add_argument('actions', type=str)
    parser.add_argument('--iters', type=int, default=10)

    args = parser.parse_args()

    h = 50
    w = 50
    res = tf.constant([0.1], dtype=tf.float32, name='resolution')
    state = tf.Variable([[0.0, 0.0, 0.02, 0.1, -0.02, 0.2]], dtype=tf.float32, name='state')
    origins = tf.constant([0.0, 0.0], dtype=tf.float32, name='origins')

    optimizer = tf.keras.optimizers.Adam()
    alpha_goal = 100
    alpha_constraints = 1

    # FIXME: get rid of the need to specify the "type"
    fwd_model, _ = model_utils.load_generic_model(pathlib.Path(args.fwd_model_dir))
    classifier_model, _ = model_utils.load_generic_model(pathlib.Path(args.classifier_model_dir))
    actions = np.genfromtxt(args.actions, delimiter=',')

    T = actions.shape[0]

    goal_point = np.array([1.0, 0.0])
    goal_point_idx = 0

    path_lengths = []
    paths = []

    params = {
        "iters": 500,
        "goal_alpha": 1000,
        "constraints_alpha": 1000,
        "action_alpha": 1
    }

    planned_path = {}

    smoother = TrajectorySmoother(fwd_model, classifier_model, params)
    smoother.smooth(actions, planned_path)

    plt.figure()
    plt.plot(path_lengths, label='path length')
    plt.xlabel("iter")
    plt.ylabel("length")
    plt.legend()

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.axis("equal")
    ax.scatter(goal_point[0], goal_point[1], label='goal', s=100, c='k')
    ax.legend()
    lines = []
    for t in range(T + 1):
        line = ax.plot(paths[0][t, :, 0], paths[0][t, :, 1])[0]
        lines.append(line)

    def update(iter):
        path = paths[iter]
        for t in range(T + 1):
            xs = path[t, :, 0]
            ys = path[t, :, 1]
            lines[t].set_data(xs, ys)

    anim = FuncAnimation(fig, update, frames=args.iters, interval=20)

    plt.show()


if __name__ == '__main__':
    main()
