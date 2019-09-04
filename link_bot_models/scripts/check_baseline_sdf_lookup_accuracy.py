#!/usr/bin/env python

import argparse

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_data.visualization import plot_rope_configuration
from link_bot_pycommon.link_bot_sdf_utils import point_to_sdf_idx
from video_prediction.datasets import dataset_utils


def show_error(rope_config, action, image, sdf_image, x, y, predicted_violated, true_violated, signed_distance, vx, vy):
    fig, ax = plt.subplots()
    arena_size = 0.5

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim([-arena_size, arena_size])
    ax.set_ylim([-arena_size, arena_size])
    ax.axis("equal")

    ax.imshow(image, extent=[-0.53, 0.53, -0.53, 0.53], zorder=0)
    ax.imshow(sdf_image, extent=[-0.5, 0.5, -0.5, 0.5], zorder=0, alpha=0.5)

    plt.quiver(rope_config[4], rope_config[5], action[0], action[1], zorder=4)

    plot_rope_configuration(ax, rope_config, linewidth=3, zorder=1, c='b')
    plt.scatter(rope_config[4], rope_config[5], c='r' if predicted_violated else 'g', s=200, zorder=2)
    plt.scatter(x, y, c='w', s=100, zorder=2)
    plt.scatter(rope_config[4], rope_config[5], c='r' if true_violated else 'g', marker='*', s=150, zorder=3, linewidths=0.1,
                edgecolors='k')
    plt.title("{} {:.3f},{:.3f} {:.3f},{:.3f}".format(signed_distance, action[0], action[1], vx, vy))
    plt.show()


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=1000)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('indir')
    parser.add_argument('dataset_hparams_dict')
    parser.add_argument('--distance-threshold', type=float, default=0.02)
    parser.add_argument('--cheat', action='store_true')
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    np.random.seed(0)
    tf.random.set_random_seed(0)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.1))
    gpf.reset_default_session(config=config)
    sess = gpf.get_default_session()

    dataset, train_inputs, steps_per_epoch = dataset_utils.get_inputs(args.indir,
                                                                      'link_bot_video',
                                                                      args.dataset_hparams_dict,
                                                                      'sequence_length=100',
                                                                      mode='train',
                                                                      epochs=1,
                                                                      seed=0,
                                                                      batch_size=1,
                                                                      shuffle=False)

    incorrect = 0
    correct = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    while True:
        try:
            data = sess.run(train_inputs)
        except tf.errors.OutOfRangeError:
            break

        rope_configurations = data['rope_configurations'].squeeze()
        sdfs = data['sdf'].squeeze()
        resolutions = data['sdf_resolution'].squeeze()
        images = data['images'].squeeze()
        origins = data['sdf_origin'].squeeze()
        constraints = data['constraints'].squeeze()
        actions = data['actions'].squeeze()
        post_action_velocities = data['post_action_velocity'].squeeze()

        zipped = zip(constraints, sdfs, rope_configurations, actions, resolutions, origins, images, post_action_velocities)
        for true_violated, sdf, rope_config, action, resolution, origin, image, vel in zipped:
            sdf_image = np.flipud(sdf.T) > 0
            x = rope_config[4]
            y = rope_config[5]
            if args.cheat:
                dx = action[0] * dataset.hparams.dt
                dy = action[1] * dataset.hparams.dt
                x += dx
                y += dy
                x = min(max(x, -0.48), 0.48)
                y = min(max(y, -0.48), 0.48)
            row, col = point_to_sdf_idx(x, y, resolution=resolution, origin=origin)
            signed_distance = sdf[row, col]
            predicted_violated = signed_distance < args.distance_threshold

            if predicted_violated:
                if true_violated:
                    correct += 1
                    tp += 1
                else:
                    incorrect += 1
                    fp += 1
                    if args.show:
                        show_error(rope_config, action, image, sdf_image, x, y, predicted_violated, true_violated, signed_distance,
                                   vel[0], vel[1])
            else:
                if true_violated:
                    incorrect += 1
                    fn += 1
                    # if args.show:
                    #     show_error(rope_config, action, image, sdf_image, x, y, predicted_violated, true_violated, signed_distance,
                    #                vel[0], vel[1])
                else:
                    correct += 1
                    tn += 1

    accuracy = correct / (correct + incorrect)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("accuracy: {:5.3f}".format(accuracy))
    print("precision: {:5.3f}".format(precision))
    print("recall: {:5.3f}".format(recall))


if __name__ == '__main__':
    main()
