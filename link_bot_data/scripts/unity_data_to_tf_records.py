#!/usr/bin/env python
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from PIL import Image
from link_bot_data import video_prediction_dataset_utils

np.set_printoptions(suppress=True, precision=4, linewidth=250)
opts = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
conf = tf.ConfigProto(gpu_options=opts)
tf.enable_eager_execution(config=conf)


def generate_tf_record(dir_name, args):
    data_path = os.path.join(dir_name, "gripper_data.bytes")
    data_f = open(data_path, "rb")
    num_trajs = int.from_bytes(data_f.read(4), byteorder='little')
    steps_per_traj = int.from_bytes(data_f.read(4), byteorder='little')
    states = np.fromfile(data_f, dtype='<f4')
    states = states.reshape(num_trajs, steps_per_traj, -1)

    actions_path = os.path.join(dir_name, "gripper_actions.bytes")
    actions_f = open(actions_path, "rb")
    num_trajs = int.from_bytes(actions_f.read(4), byteorder='little')
    steps_per_traj = int.from_bytes(actions_f.read(4), byteorder='little')
    actions = np.fromfile(actions_f, dtype='<f4')
    actions = actions.reshape(num_trajs, steps_per_traj, -1)

    expected_n_trajs = 256
    assert num_trajs == expected_n_trajs, "Number of trajectories per file must be {}".format(expected_n_trajs)

    image_bytes = np.ndarray((num_trajs, steps_per_traj), object)
    image_filenames = glob.glob(os.path.join(dir_name, "*.png"))
    sorted_image_filenames = sorted(image_filenames, key=lambda x: int(os.path.basename(x)[:-4]))
    i = 0
    for image_filename in sorted_image_filenames:
        if "png" in image_filename:
            traj_idx = i // steps_per_traj
            t = i % steps_per_traj
            img = Image.open(image_filename)
            image_array = np.array(img)
            image_bytes[traj_idx, t] = image_array[:, :, :3].tobytes()
            i += 1

    dataset = tf.data.Dataset.from_tensor_slices((image_bytes, states, actions))

    serialized_dataset = dataset.map(video_prediction_dataset_utils.tf_serialize_example)

    filename = '{}.tfrecords'.format(dir_name)
    full_filename = os.path.join(args.out_dir, filename)
    writer = tf.data.experimental.TFRecordWriter(full_filename)
    print("saving {}".format(filename))
    writer.write(serialized_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_glob")
    parser.add_argument("out_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-processes", type=int, default=4)

    args = parser.parse_args()

    np.random.seed(args.seed)

    folders = glob.glob(args.in_glob)
    num_tfrecords = len(folders)

    process_arguments = zip(folders, [args] * num_tfrecords)
    with Pool(args.num_processes) as p:
        p.starmap(generate_tf_record, process_arguments)


if __name__ == '__main__':
    main()
