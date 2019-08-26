#!/usr/bin/env python
import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

from video_prediction import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('dataset_hparams_dict')
    parser.add_argument('mode', choices=['train', 'val', 'test'])
    parser.add_argument('--dataset-hparams')

    np.random.seed(0)
    tf.random.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.FATAL)

    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4, linewidth=250)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    sess = tf.Session(config=config)

    VideoDataset = datasets.get_dataset_class('link_bot_video')
    with open(args.dataset_hparams_dict, 'r') as hparams_f:
        hparams_dict = json.loads(hparams_f.read())
    hparams_dict['sequence_length'] = 100
    dataset = VideoDataset(args.input_dir, mode=args.mode, seed=1, num_epochs=1, hparams_dict=hparams_dict,
                           hparams=args.dataset_hparams)

    inputs = dataset.make_batch(1, shuffle=False)
    traj_idx = 0
    while True:
        try:
            outputs = sess.run(inputs)
        except tf.errors.OutOfRangeError:
            break
        rope_configurations_traj = outputs['rope_configurations']
        images_traj = outputs['images']
        rope_configuration_traj = np.squeeze(rope_configurations_traj)
        image_traj = np.squeeze(images_traj)
        for t, (rope_configuration, image) in enumerate(zip(rope_configuration_traj, image_traj)):
            if np.any(rope_configuration > 0.5) or np.any(rope_configuration < -0.5):
                bad_file_idx = int(traj_idx / 128)
                bad_filename = dataset.filenames[bad_file_idx]
                print(bad_filename)
                # plt.imshow(image)
                # plt.title('{}\n{}, {}\n{}'.format(bad_filename, traj_idx, t, rope_configuration))
                # plt.show()
        traj_idx += 1


if __name__ == '__main__':
    main()
