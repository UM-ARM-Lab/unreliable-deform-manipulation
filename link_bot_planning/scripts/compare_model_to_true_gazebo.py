#!/usr/bin/env python

import argparse
import pathlib
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import rospy
import tensorflow as tf
from matplotlib import animation

from link_bot_data.visualization import plottable_rope_configuration
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_gazebo.srv import LinkBotStateRequest, WorldControlRequest
from link_bot_planning import model_utils

tf.enable_eager_execution()


def visualize(args, predicted_points, actual_traj):
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

        actual_rope_config = actual_traj[t]
        actual_xs, actual_ys = plottable_rope_configuration(actual_rope_config)
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

    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rospy.init_node('compare_to_true_gazebo')

    # Start Services
    services = gazebo_utils.GazeboServices()
    services.reset_gazebo_environment(reset_model_poses=False)

    state_req = LinkBotStateRequest()
    state = services.get_state.call(state_req)
    initial_rope_configuration = np.array([[p.x, p.y] for p in state.points]).flatten()

    actions = np.genfromtxt(args.actions, delimiter=',')

    fwd_model = model_utils.load_generic_model(args.model_dir, args.model_type)
    dt = fwd_model.dt

    predicted_points = fwd_model.predict(np.expand_dims(initial_rope_configuration, axis=0), np.expand_dims(actions, axis=0))
    predicted_points = predicted_points[0]

    # execute actions in gazebo
    action_msg = LinkBotVelocityAction()
    actual_traj = [initial_rope_configuration]
    for action in actions:
        # publish the command
        action_msg.gripper1_velocity.x = action[0]
        action_msg.gripper1_velocity.y = action[1]
        services.velocity_action_pub.publish(action_msg)

        step = WorldControlRequest()
        step.steps = int(dt / 0.001)  # assuming 0.001s of simulation time per step
        services.world_control.call(step)  # this will block until stepping is complete

        state = services.get_state(state_req)
        actual_config = gazebo_utils.points_to_config(state.points)
        actual_traj.append(actual_config)

    actual_traj = np.array(actual_traj)

    visualize(args, predicted_points, actual_traj)


if __name__ == '__main__':
    main()
