#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_planning import model_utils, classifier_utils
from link_bot_planning.link_bot_scenario import LinkBotScenario
from link_bot_planning.trajectory_smoother import TrajectorySmoother
from link_bot_pycommon.args import my_formatter

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)



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

    goal = np.array([-1.25, 1.667])
    goal_idx = 0
    goal_subspace_feature_name = 'link_bot'

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
        'link_bot': np.array([0.0, 0.0,
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

    experiment_scenario = LinkBotScenario()

    #####################################################
    # Interactive Visualization - Do the Actual Smoothing
    #####################################################
    smoother = TrajectorySmoother(fwd_model=fwd_model,
                                  classifier_model=classifier_model,
                                  experiment_scenario=experiment_scenario,
                                  params=params,
                                  verbose=2,
                                  )

    actions = tf.Variable(actions, dtype=tf.float32, name='controls', trainable=True)

    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Path")
    experiment_scenario.plot_goal(axes[0], goal, 'g')
    experiment_scenario.plot_state_simple(axes[0], start_states, 'r')
    axes[0].set_xlim([-2.5,2.5])
    axes[0].set_ylim([-2.5, 2.5])
    axes[0].legend()

    losses_line = axes[1].plot([], label='total loss')[0]
    length_losses_line = axes[1].plot([], label='length loss')[0]
    goal_losses_line = axes[1].plot([], label='goal loss')[0]
    constraint_losses_line = axes[1].plot([], label='constraint loss')[0]
    action_losses_line = axes[1].plot([], label='action loss')[0]
    axes[1].set_xlabel("iter")
    axes[1].set_ylabel("loss")
    axes[1].set_yscale("log")
    axes[1].set_title("Losses")
    axes[1].legend()

    iters = []
    losses = []
    length_losses = []
    goal_losses = []
    constraints_losses = []
    action_losses = []
    step_times = []

    artists = []
    for t in range(T + 1):
        artist = experiment_scenario.plot_state(axes[0], planned_path[t], 'b')
        artists.append(artist)

    def update(iter):
        nonlocal actions
        t0 = perf_counter()
        actions, predictions, step_losses = smoother.step(full_env=full_env,
                                                          full_env_origin=full_env_origin,
                                                          goal=goal,
                                                          res=res,
                                                          actions=actions,
                                                          planned_path=planned_path)
        dt = perf_counter() - t0
        step_times.append(dt)

        length_loss, goal_loss, constraint_loss, action_loss, loss = step_losses
        loss = loss.numpy()
        length_loss = length_loss.numpy()
        goal_loss = goal_loss.numpy()
        constraint_loss = constraint_loss.numpy()
        action_loss = action_loss.numpy()

        for t in range(T + 1):
            experiment_scenario.update_artist(artists[t], predictions[t])

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

        # axes[0].relim()
        # axes[0].autoscale_view()
        axes[1].relim()
        axes[1].autoscale_view()

    anim = FuncAnimation(fig, update, frames=args.iters, interval=1, repeat=False)
    anim.save("smoothing_animation.gif", writer='imagemagick')
    print("Mean step time: {:8.5f}s, {:8.5f}".format(np.mean(step_times), np.std(step_times)))
    plt.show()


if __name__ == '__main__':
    main()
