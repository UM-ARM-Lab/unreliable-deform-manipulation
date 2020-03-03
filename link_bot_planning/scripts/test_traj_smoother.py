#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import pathlib

import numpy as np
import tensorflow as tf

from link_bot_planning import model_utils
from link_bot_pycommon.args import my_formatter
from moonshine.raster_points_layer import differentiable_raster

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
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('fwd_model_dir', type=pathlib.Path)
    parser.add_argument('fwd_model_type', type=str)
    parser.add_argument('actions', type=str)

    args = parser.parse_args()

    h = 50
    w = 50
    res = tf.constant([0.1], dtype=tf.float32, name='resolution')
    state = tf.Variable([[0.19, 1.51, 2.12, 1.62, -0.01, 0.01]], dtype=tf.float32, name='state')
    origins = tf.constant([0.0, 0.0], dtype=tf.float32, name='origins')

    optimizer = tf.compat.v1.train.AdamOptimizer()
    alpha_goal = 1
    alpha_constraints = 1

    # fwd_model, _ = model_utils.load_generic_model(pathlib.Path(args.fwd_model_dir), args.fwd_model_type)
    actions = tf.Variable(np.genfromtxt(args.actions, delimiter=','), dtype=tf.float32, name='actions', trainable=True)

    T = actions.shape[0]

    goal_point = np.array([1.0, 0.0])
    goal_point_idx = 0

    path_lengths = []
    for i in range(100):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            tape.watch(state)
            # image = differentiable_raster(state, res, origins, h, w)
            # input_dict = {
            #     fwd_model.net.state_feature: state,
            #     'action': actions,
            # }
            # predictions = fwd_model.net(input_dict)
            predictions = mock_fwd_model(state, actions)
            predicted_points = tf.reshape(predictions, [T + 1, -1, 2])
            deltas = predicted_points[1:] - predicted_points[:-1]
            final_target_point_pred = predicted_points[-1, goal_point_idx]
            goal_loss = tf.square(final_target_point_pred - goal_point)
            distances = tf.linalg.norm(deltas)
            length_loss = tf.reduce_sum(distances)
            constraint_loss = 0  # TODO:
            # loss = length_loss + alpha_goal * goal_loss + alpha_constraints * constraint_loss
            loss = length_loss

        variables = [actions]
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        path_lengths.append(length_loss)

    plt.plot(path_lengths, label='path length')
    plt.xlabel("iter")
    plt.ylabel("length")
    plt.show()


if __name__ == '__main__':
    main()
