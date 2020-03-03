#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_planning import model_utils, classifier_utils
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
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    plt.style.use("slides")

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('fwd_model_dir', type=pathlib.Path)
    parser.add_argument('classifier_model_dir', type=pathlib.Path)
    parser.add_argument('actions', type=str)
    parser.add_argument('--iters', type=int, default=500)

    args = parser.parse_args()

    ######################
    # Setup
    ######################
    res = 0.1
    full_env = np.zeros([200, 200], dtype=np.float32)
    full_env[20:50, 40:100] = 1.0
    full_env_origin = np.array([100, 100])

    fwd_model, _ = model_utils.load_generic_model(pathlib.Path(args.fwd_model_dir))
    classifier_model = classifier_utils.load_generic_model(pathlib.Path(args.classifier_model_dir))
    actions = np.genfromtxt(args.actions, delimiter=',')

    T = actions.shape[0]

    goal_point = np.array([2.0, 0.0])
    goal_point_idx = 0
    goal_subspace_name = 'link_bot'

    params = {
        "iters": args.iters,
        "goal_alpha": 1000,
        "constraints_alpha": 1,
        "action_alpha": 1
    }

    ###########################
    # Construct initial path
    ###########################
    start_states = {
        'state/link_bot': np.array([0.0, 0.0,
                                    0.05, 0.0,
                                    0.10, 0.0,
                                    0.15, 0.0,
                                    0.20, 0.0,
                                    0.25, 0.0,
                                    0.30, 0.0,
                                    0.35, 0.0,
                                    0.40, 0.0,
                                    0.45, 0.0,
                                    0.55, 0.0,
                                    ])
    }

    planned_path = fwd_model.propagate_differentiable(full_env=full_env,
                                                      full_env_origin=full_env_origin,
                                                      res=res,
                                                      start_states=start_states,
                                                      actions=actions)

    #####################################################
    # Interactive Visualization - Do the Actual Smoothing
    #####################################################
    smoother = TrajectorySmoother(fwd_model=fwd_model,
                                  classifier_model=classifier_model,
                                  params=params,
                                  goal_point_idx=goal_point_idx,
                                  goal_subspace_name=goal_subspace_name)

    actions = tf.Variable(actions, dtype=tf.float32, name='controls', trainable=True)

    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Path")
    axes[0].scatter(goal_point[0], goal_point[1], label='goal', s=50, c='k')
    axes[0].scatter(start_states['state/link_bot'][0], start_states['state/link_bot'][1], label='start', s=50, c='r')
    axes[0].set_xlim([-3, 3])
    axes[0].set_ylim([-3, 3])
    axes[0].axis("equal")
    axes[0].legend()
    path_lines = []
    for t in range(T + 1):
        line = axes[0].plot([], [])[0]
        path_lines.append(line)

    losses_line = axes[1].plot([], label='total loss')[0]
    length_losses_line = axes[1].plot([], label='length loss')[0]
    goal_losses_line = axes[1].plot([], label='goal loss')[0]
    constraint_losses_line = axes[1].plot([], label='constraint loss')[0]
    action_losses_line = axes[1].plot([], label='action loss')[0]
    axes[1].set_xlabel("iter")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Losses")
    axes[1].legend()

    iters = []
    losses = []
    length_losses = []
    goal_losses = []
    constraints_losses = []
    action_losses = []

    def update(iter):
        nonlocal actions
        step_result = smoother.step(full_env=full_env,
                                    full_env_origin=full_env_origin,
                                    goal_point=goal_point,
                                    res=res,
                                    actions=actions,
                                    planned_path=planned_path)

        actions, predictions, length_loss, goal_loss, constraint_loss, action_loss, loss = step_result
        loss = loss.numpy()
        length_loss = length_loss.numpy()
        goal_loss = goal_loss.numpy()
        constraint_loss = constraint_loss.numpy()
        action_loss = action_loss.numpy()

        predicted_points = tf.reshape(predictions['state/link_bot'], [T + 1, -1, 2]).numpy()
        for t in range(T + 1):
            xs = predicted_points[t, :, 0]
            ys = predicted_points[t, :, 1]
            path_lines[t].set_data(xs, ys)

        iters.append(iter)
        losses.append(loss)
        length_losses.append(length_loss)
        goal_losses.append(goal_loss)
        constraints_losses.append(constraint_loss)
        action_losses.append(action_loss)

        losses_line.set_data(iters, losses)
        length_losses_line.set_data(iters, length_losses)
        goal_losses_line.set_data(iters, goal_losses)
        constraint_losses_line.set_data(iters, constraints_losses)
        action_losses_line.set_data(iters, action_losses)

        axes[0].relim()
        axes[0].autoscale_view()
        axes[1].relim()
        axes[1].autoscale_view()

    anim = FuncAnimation(fig, update, frames=args.iters, interval=0, repeat=False)

    plt.show()


if __name__ == '__main__':
    main()
