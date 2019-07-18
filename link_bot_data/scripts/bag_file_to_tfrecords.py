#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import os

import numpy as np
import rosbag
import tensorflow as tf
from colorama import Fore

np.set_printoptions(suppress=True, precision=4, linewidth=250)
opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=False)
conf = tf.ConfigProto(gpu_options=opts)
tf.enable_eager_execution(config=conf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_file")
    parser.add_argument("out_dir")

    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        print(Fore.YELLOW + "Output directory {} does not exist. Aborting".format(args.out_dir) + Fore.RESET)
        return

    topics = [
        "left_gripper/wrench",
        "right_gripper/wrench",
        "kinect_victor_head/hd/image_color/compressed",
        "tf",
    ]

    bag = rosbag.Bag(args.bag_file)
    trajs_per_tfrecords = 128
    for topic, msg, t in bag.read_messages(topics=topics):
        test_images = None
        test_states = None
        test_actions = None

        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_states, test_actions))

        serialized_test_dataset = test_dataset.map(tf_serialize_example)

        filename = 'traj_{}_to_{}.tfrecords'.format(traj_idx_start, traj_idx_start + trajs_per_tfrecords)
        print("saving {}".format(filename))
        full_filename = os.path.join(args.out_dir, filename)
        writer = tf.data.experimental.TFRecordWriter(full_filename)
        writer.write(serialized_test_dataset)

    bag.close()


if __name__ == '__main__':
    main()
