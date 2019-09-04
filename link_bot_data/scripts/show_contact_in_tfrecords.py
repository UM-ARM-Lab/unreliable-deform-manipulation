#!/usr/bin/env python
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_pycommon.link_bot_sdf_utils import sdf_bounds
from video_prediction.datasets import LinkBotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="directory of tfrecords")
    parser.add_argument("dataset_hparams_dict", type=str, help="json file of hyperparameters")
    parser.add_argument("n_examples", type=int)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=['test', 'val', 'train'], default='train')
    parser.add_argument("--dataset-hparams", type=str, help="a string of comma separated list of dataset hyperparameters")

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    dataset_hparams_dict = json.load(open(args.dataset_hparams_dict, 'r'))
    dataset = LinkBotDataset(args.input_dir,
                             mode=args.mode,
                             num_epochs=1,
                             seed=args.seed,
                             hparams_dict=dataset_hparams_dict,
                             hparams=args.dataset_hparams)

    inputs = dataset.make_batch(1, shuffle=False)

    negative_count = 0
    positive_count = 0

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    while negative_count != args.n_examples or positive_count != args.n_examples:
        try:
            input_results = sess.run(inputs)
        except tf.errors.OutOfRangeError:
            break
        sdf = input_results['sdf'].squeeze()
        image = input_results['images'].squeeze()
        res = input_results['sdf_resolution'].squeeze()
        origin = input_results['sdf_origin'].squeeze()
        rope_config = input_results['rope_configurations'].squeeze()
        velocity = input_results['post_action_velocity'].squeeze()
        action = input_results['actions'].squeeze()
        vx, vy = action
        constraint = bool(input_results['constraints'].squeeze())
        extent = sdf_bounds(sdf, res, origin)
        occupancy_image = np.flipud(input_results['sdf'].squeeze().T) > 0

        color = 'r' if constraint else 'g'
        if constraint:
            if positive_count == args.n_examples:
                continue
            positive_count += 1
        else:
            if negative_count == args.n_examples:
                continue
            negative_count += 1

        plt.figure()
        # plt.imshow(image, extent=extent, cmap='jet')
        plt.imshow(occupancy_image, extent=extent, cmap='jet')
        xs = [rope_config[0], rope_config[2], rope_config[4]]
        ys = [rope_config[1], rope_config[3], rope_config[5]]
        plt.plot(xs, ys, zorder=2, c='k')
        plt.scatter(rope_config[4], rope_config[5], zorder=3, c=color)
        plt.quiver(rope_config[4], rope_config[5], vx, vy, zorder=2, color='orange', scale=0.75, alpha=0.5)
        plt.title("commanded: {:.3f},{:.3f} m/s, measured {:.3f},{:.3f} m/s".format(action[0], action[1], velocity[0], velocity[1]))
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
    plt.show()


if __name__ == '__main__':
    main()
