#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from colorama import Fore

np.set_printoptions(suppress=True, precision=4, linewidth=250)
opts = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
conf = tf.ConfigProto(gpu_options=opts)
tf.enable_eager_execution(config=conf)

from link_bot_data import video_prediction_dataset_utils


def generate_tf_record(record_idx, trajs_per_tfrecords, args):
    env_size = 1.0
    block_size = 0.125
    traj_length = 60
    image_size = 64
    state_dim = 2
    action_dim = 2
    m2pix = image_size / env_size
    table_x_low = -env_size / 2 + block_size / 2
    table_x_high = env_size / 2 - block_size / 2
    table_y_low = -env_size / 2 + block_size / 2
    table_y_high = env_size / 2 - block_size / 2
    velocity_low = 0
    velocity_high = 0.3
    dt = 0.1
    image = np.zeros((image_size, image_size, 3), np.uint8)

    def point_to_block_slice(state):
        block_x_low = int((state[0] - block_size / 2 + env_size / 2) * m2pix)
        block_x_high = int((state[0] + block_size / 2 + env_size / 2) * m2pix)
        block_y_low = int((state[1] - block_size / 2 + env_size / 2) * m2pix)
        block_y_high = int((state[1] + block_size / 2 + env_size / 2) * m2pix)
        return slice(block_y_low, block_y_high), slice(block_x_low, block_x_high), 0

    test_image_bytes = np.ndarray((trajs_per_tfrecords, traj_length), object)
    test_states = np.ndarray((trajs_per_tfrecords, traj_length, state_dim), np.float32)
    test_actions = np.ndarray((trajs_per_tfrecords, traj_length, action_dim), np.float32)
    for traj_idx in range(trajs_per_tfrecords):
        x = np.random.uniform(table_x_low, table_x_high)
        y = np.random.uniform(table_y_low, table_y_high)
        state = np.array([x, y])
        action = np.random.uniform(velocity_low, velocity_high, size=2)
        for t in range(traj_length):
            image.fill(0)
            block_slice = point_to_block_slice(state)
            image[block_slice] = 255
            test_image_bytes[traj_idx, t] = image.tobytes()
            test_states[traj_idx, t] = state
            test_actions[traj_idx, t] = action

            # Motion update of the block
            next_state = state + action * dt
            # Negate the action if the block is going out of bounds, i.e. bounce off the borders
            if next_state[0] > table_x_high or next_state[0] < table_x_low:
                action[0] = -action[0]
            elif next_state[1] > table_x_high or next_state[1] < table_x_low:
                action[1] = -action[1]
            state = state + action * dt

    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_bytes, test_states, test_actions))

    serialized_test_dataset = test_dataset.map(video_prediction_dataset_utils.tf_serialize_example)

    traj_idx_start = record_idx * trajs_per_tfrecords
    filename = 'traj_{}_to_{}.tfrecords'.format(traj_idx_start, traj_idx_start + trajs_per_tfrecords - 1)
    print("saving {}".format(filename))
    full_filename = os.path.join(args.out_dir, filename)
    writer = tf.data.experimental.TFRecordWriter(full_filename)
    writer.write(serialized_test_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-processes", type=int, default=4)

    args = parser.parse_args()

    np.random.seed(args.seed)

    if not os.path.isdir(args.out_dir):
        print(Fore.YELLOW + "Output directory {} does not exist. Aborting".format(args.out_dir) + Fore.RESET)
        return

    num_examples = 1000
    # If you change this number the video_prediction code breaks
    trajs_per_tfrecords = 256
    num_tfrecords = num_examples // trajs_per_tfrecords

    process_arguments = zip(range(num_tfrecords), [trajs_per_tfrecords] * num_tfrecords, [args] * num_tfrecords)
    for args in process_arguments:
        generate_tf_record(*args)
    # with Pool(args.num_processes) as p:
    #     p.starmap(generate_tf_record, process_arguments)


if __name__ == '__main__':
    main()
