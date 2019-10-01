#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

from video_prediction.datasets import dataset_utils

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


def test(args):
    if args.dataset_hparams_dict:
        dataset_hparams_dict = json.load(open(args.dataset_hparams_dict, 'r'))
    else:
        dataset_hparams_dict = json.load(open(args.input_dir / 'hparams.json', 'r'))

    ###############
    # Datasets
    ###############
    dataset_hparams_dict['sequence_length'] = args.sequence_length
    dt = dataset_hparams_dict['dt']
    dataset, tf_dataset = dataset_utils.get_dataset(args.input_dir,
                                                    'state_space',
                                                    dataset_hparams_dict,
                                                    args.dataset_hparams,
                                                    shuffle=False,
                                                    mode=args.mode,
                                                    epochs=1,
                                                    seed=0,
                                                    batch_size=1)

    mses = []
    head_mses = []
    mid_mses = []
    tail_mses = []
    for x, y in tf_dataset:
        states = x['states'].numpy()
        actions = x['actions'].numpy().squeeze()
        points = np.reshape(states, [-1, 3, 2])
        s_0 = points[0]
        s_t = s_0
        prediction = [s_0]
        for action in actions:
            s_t = s_t + np.reshape(np.tile(np.eye(2), [3, 1]) @ action, [3, 2]) * dt
            prediction.append(s_t)
        prediction = np.array(prediction)
        true = y['output_states'].numpy().squeeze().reshape([-1, 3, 2])
        mse = np.mean((prediction - true)**2)
        head_mse = np.mean((prediction[:, 2] - true[:, 2])**2)
        mid_mse = np.mean((prediction[:, 1] - true[:, 1])**2)
        tail_mse = np.mean((prediction[:, 0] - true[:, 0])**2)
        head_mses.append(head_mse)
        mid_mses.append(mid_mse)
        tail_mses.append(tail_mse)
        mses.append(mse)
    print("head MSE:    {:8.5f}m".format(np.mean(head_mses)))
    print("mid MSE:     {:8.5f}m".format(np.mean(mid_mses)))
    print("tail MSE:    {:8.5f}m".format(np.mean(tail_mses)))
    print("overall MSE: {:8.5f}m".format(np.mean(mses)))


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    parser = subparsers.add_parser('train')
    parser.add_argument('input_dir', type=pathlib.Path)
    parser.add_argument('--dataset-hparams-dict', type=pathlib.Path)
    parser.add_argument('--dataset-hparams', type=str)
    parser.add_argument('--sequence-length', type=int, default=10)
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')

    args = parser.parse_args()

    test(args)


if __name__ == '__main__':
    main()
