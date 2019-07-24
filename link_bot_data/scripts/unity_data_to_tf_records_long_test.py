#!/usr/bin/env python
import pathlib
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

    total_size = num_trajs * steps_per_traj
    assert total_size % args.seq_length == 0
    num_trajs = int(total_size / args.seq_length)
    steps_per_traj = int(total_size / num_trajs)
    print(num_trajs, steps_per_traj)

    states = np.fromfile(data_f, dtype='<f4')
    states = states.reshape(num_trajs, steps_per_traj, -1)

    actions_path = os.path.join(dir_name, "gripper_actions.bytes")
    actions_f = open(actions_path, "rb")
    _ = int.from_bytes(actions_f.read(4), byteorder='little')
    _ = int.from_bytes(actions_f.read(4), byteorder='little')
    actions = np.fromfile(actions_f, dtype='<f4')
    actions = actions.reshape(num_trajs, steps_per_traj, -1)

    image_bytes = np.ndarray((num_trajs, steps_per_traj), object)
    zero_image = np.zeros((64, 64, 3), np.uint8)
    image_filenames = glob.glob(os.path.join(dir_name, "*.png"))
    sorted_image_filenames = sorted(image_filenames, key=lambda x: int(os.path.basename(x)[:-4]))
    i = 0
    for image_filename in sorted_image_filenames:
        if "png" in image_filename:
            traj_idx = i // steps_per_traj
            t = i % steps_per_traj
            img = Image.open(image_filename)
            image_array = np.array(img)
            if t == 0:
                image_bytes[traj_idx, t] = image_array[:, :, :3].tobytes()
            else:
                image_bytes[traj_idx, t] = zero_image.tobytes()
            i += 1

    print(image_bytes.shape, states.shape, actions.shape)
    dataset = tf.data.Dataset.from_tensor_slices((image_bytes, states, actions))

    serialized_dataset = dataset.map(video_prediction_dataset_utils.tf_serialize_example)

    path = pathlib.Path(dir_name)
    trajs_range_name = path.parts[-1]
    filename = '{}.tfrecords'.format(trajs_range_name)
    full_path = pathlib.Path(args.out_dir) / filename
    full_filename = str(full_path)
    writer = tf.data.experimental.TFRecordWriter(full_filename)
    print("saving {}".format(full_filename))
    writer.write(serialized_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_glob")
    parser.add_argument("out_dir")
    parser.add_argument("seq_length", type=int)
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
