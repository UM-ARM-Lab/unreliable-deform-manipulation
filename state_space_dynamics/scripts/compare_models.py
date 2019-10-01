#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

from link_bot_gaussian_process import link_bot_gp
from state_space_dynamics.locally_linear_nn import LocallyLinearNN, LocallyLinearNNWrapper
from state_space_dynamics.rigid_translation_model import RigidTranslationModel
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

    no_penalty_gp_path = args.no_penalty_gp_dir / "fwd_model"
    no_penalty_gp = link_bot_gp.LinkBotGP()
    no_penalty_gp.load(no_penalty_gp_path)

    penalty_gp_path = args.penalty_gp_dir / "fwd_model"
    penalty_gp = link_bot_gp.LinkBotGP()
    penalty_gp.load(penalty_gp_path)

    rigid_translation = RigidTranslationModel(beta=0.7, dt=dt)

    llnn = LocallyLinearNNWrapper(path)

    models = {
        'GP no penalty': no_penalty_gp,
        'GP with penalty': penalty_gp,
        'rigid-translation': rigid_translation,
        'LL-NN': llnn,
    }

    for model_name, model in models.items():
        total_errors = []
        head_errors = []
        mid_errors = []
        tail_errors = []

        for x, y in tf_dataset:
            states = x['states'].numpy()
            actions = x['actions'].numpy().squeeze()
            true = y['output_states'].numpy().squeeze().reshape([-1, 3, 2])

            prediction = model.predict(states, actions)
            error = np.linalg.norm(prediction - true, axis=2)
            total_error = np.sum(error, axis=1)
            tail_error = error[:, 0]
            mid_error = error[:, 1]
            head_error = error[:, 2]
            head_errors.append(head_error)
            mid_errors.append(mid_error)
            tail_errors.append(tail_error)
            total_errors.append(total_error)

        print("Model {}".format(model_name))
        print("head error:  {:8.4f}m".format(np.mean(head_errors)))
        print("mid error:   {:8.4f}m".format(np.mean(mid_errors)))
        print("tail error:  {:8.4f}m".format(np.mean(tail_errors)))
        print("total error: {:8.4f}m".format(np.mean(total_errors)))


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    parser = subparsers.add_parser('train')
    parser.add_argument('input_dir', type=pathlib.Path)
    parser.add_argument('no_penalty_gp_dir', type=pathlib.Path)
    parser.add_argument('penalty_gp_dir', type=pathlib.Path)
    parser.add_argument('llnn_dir', type=pathlib.Path)
    parser.add_argument('--dataset-hparams-dict', type=pathlib.Path)
    parser.add_argument('--dataset-hparams', type=str)
    parser.add_argument('--sequence-length', type=int, default=10)
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')

    args = parser.parse_args()

    test(args)


if __name__ == '__main__':
    main()
