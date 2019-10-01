#!/usr/bin/env python
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_data.visualization import plottable_rope_configuration
from link_bot_gaussian_process import link_bot_gp
from state_space_dynamics.locally_linear_nn import LocallyLinearNNWrapper
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

    llnn = LocallyLinearNNWrapper(args.llnn_dir)

    models = {
        'LL-NN': llnn,
        'GP no penalty': no_penalty_gp,
        'GP with penalty': penalty_gp,
        'rigid-translation': rigid_translation,
    }

    #######################
    # Compute error metrics
    #######################
    for model_name, model in models.items():
        total_errors = []
        head_errors = []
        mid_errors = []
        tail_errors = []

        for x, y in tf_dataset:
            states = x['states'].numpy()
            actions = x['actions'].numpy().squeeze()
            true_points = y['output_states'].numpy().squeeze().reshape([-1, 3, 2])

            predicted_points = model.predict(states, actions)
            error = np.linalg.norm(predicted_points - true_points, axis=2)
            total_error = np.sum(error, axis=1)
            tail_error = error[:, 0]
            mid_error = error[:, 1]
            head_error = error[:, 2]
            head_errors.append(head_error)
            mid_errors.append(mid_error)
            tail_errors.append(tail_error)
            total_errors.append(total_error)

        print("Model: {}".format(model_name))
        print("head error:  {:8.4f}m".format(np.mean(head_errors)))
        print("mid error:   {:8.4f}m".format(np.mean(mid_errors)))
        print("tail error:  {:8.4f}m".format(np.mean(tail_errors)))
        print("total error: {:8.4f}m".format(np.mean(total_errors)))

    #######################
    # Visualize some examples
    #######################
    if args.no_plot:
        return

    for example_idx, (x, y) in enumerate(tf_dataset.take(args.n_examples)):
        states = x['states'].numpy()
        actions = x['actions'].numpy().squeeze()
        true_configuration_trajectory = y['output_states'].numpy().squeeze()

        fig, ax = plt.subplots()
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        plt.title(example_idx)

        time_text_handle = plt.text(5, 8, 't=0', fontdict={'color': 'white', 'size': 5}, bbox=dict(facecolor='black', alpha=0.5))


        # our plotting utilities assume (6) not (3,2)
        configuration_trajectories_by_model = {}
        for model_name, model in models.items():
            predicted_points = model.predict(states, actions)
            configuration_trajectory = predicted_points.reshape([-1, 6])
            configuration_trajectories_by_model[model_name] = configuration_trajectory
        configuration_trajectories_by_model['true'] = true_configuration_trajectory

        # create all the necessary plotting handles
        handles = {}
        for model_name, model in configuration_trajectories_by_model.items():
            handles[model_name] = {}
            handles[model_name]['line'] = plt.plot([], [], alpha=0.5, label=model_name)[0]
            handles[model_name]['scatt'] = plt.scatter([], [])

        def update(t):
            for model_name, _configuration_trajectory in configuration_trajectories_by_model.items():
                configuration = _configuration_trajectory[t]
                xs, ys = plottable_rope_configuration(configuration)
                handles[model_name]['line'].set_xdata(xs)
                handles[model_name]['line'].set_ydata(ys)
                scatt_coords = np.vstack((xs, ys)).T
                handles[model_name]['scatt'].set_offsets(scatt_coords)
            time_text_handle.set_text("t={}".format(t))

        plt.legend()
        plt.tight_layout()

        _ = FuncAnimation(fig, update, frames=args.sequence_length, interval=250)
        plt.show()


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
    parser.add_argument('--n-examples', type=int, default=10)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test')

    args = parser.parse_args()

    test(args)


if __name__ == '__main__':
    main()
