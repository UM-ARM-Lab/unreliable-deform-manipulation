#!/usr/bin/env python
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from link_bot_classifiers.rnn_recovery_model import RNNRecoveryModelWrapper
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature, add_predicted
from link_bot_data.recovery_dataset import RecoveryDataset
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import paths_from_json
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def sample_starting_points(n_samples):
    mu = tf.constant([[0.3, 0.0], [-0.6, 0.5]], dtype=tf.float32)
    sigma = tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)
    gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=tf.constant([0.5, 0.5])),
                               components_distribution=tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma))
    points = gm.sample(n_samples)
    return tf.cast(points, tf.float32)


def make_points(starting_points, actions):
    starting_points = tf.expand_dims(starting_points, axis=1)
    points = tf.cast(tf.math.cumsum(tf.concat([starting_points, actions], axis=1), axis=1), tf.float32)
    return points


def sample_actions_from_true_distribution(n_samples, n_actions, starting_points):
    alpha = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), [n_samples, 1])
    sigma = np.tile(np.array([[[0.01, 0.01], [0.01, 0.01]]], dtype=np.float32), [n_samples, 1, 1])

    def conditional_probability(_starting_points, _actions):
        if len(_actions) == 0:
            _mu_t = []
            for _starting_point in _starting_points:
                _mu_t_i = [[-0.1, 0.0], [0.1, 0.0]]
                # if _starting_point[0] > 0:
                #     _mu_t_i = [[-0.1, 0.0], [0.1, 0.0]]
                # else:
                #     _mu_t_i = [[-0.05, 0.15], [0.05, 0.15]]
                _mu_t.append(_mu_t_i)
            _mu_t = np.array(_mu_t, dtype=np.float32)
        else:
            _mu_t = []
            _last_actions = _actions[-1]
            for _starting_point, _action in zip(_starting_points, _last_actions):
                if _action[0] < 0:
                    _mu_t_i = [[-0.1, 0.0], [-0.1, 0.0]]
                else:
                    _mu_t_i = [[0.1, 0.0], [0.1, 0.0]]
                # if _starting_point[0] > 0:
                #     if _action[0] < 0:
                #         _mu_t_i = [[-0.1, 0.0], [-0.1, 0.0]]
                #     else:
                #         _mu_t_i = [[0.1, 0.0], [0.1, 0.0]]
                # else:
                #     if _action[0] < 0:
                #         _mu_t_i = [[-0.02, 0.15], [-0.02, 0.15]]
                #     else:
                #         _mu_t_i = [[0.02, 0.15], [0.02, 0.15]]
                _mu_t.append(_mu_t_i)
            _mu_t = np.array(_mu_t, dtype=np.float32)
        return _mu_t

    actions = []
    for t in range(n_actions):
        mu_t = conditional_probability(starting_points, actions)
        gm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=alpha),
                                   components_distribution=tfd.MultivariateNormalDiag(loc=mu_t, scale_diag=sigma))
        actions_t = gm.sample()
        actions.append(actions_t)
    actions = tf.stack(actions, axis=1)
    points = make_points(starting_points, actions)
    return actions, points


def generate_examples():
    examples = []
    n_samples = 1000
    n_actions = 5
    starting_points = sample_starting_points(n_samples)
    actions, points = sample_actions_from_true_distribution(n_samples, n_actions, starting_points)
    for example_idx, (points_i, actions_i) in enumerate(zip(points, actions)):
        example = {
            'full_env/env': np.zeros([200, 200], dtype=np.float32),
            'full_env/origin': [0, 0],
            'full_env/extent': [-1, 1, -1, 1],
            'full_env/res': 0.01,
            'traj_idx': 0,
            'prediction_start_t': 0,
            'classifier_start_t': 0,
            'classifier_end_t': 0,
            add_predicted('link_bot'): points_i,
            'action': actions_i,
            'is_close': np.ones([n_actions + 1], dtype=np.float32),
            'mask': np.ones([n_actions + 1], dtype=np.float32),
        }

        features = {k: float_tensor_to_bytes_feature(v) for k, v in example.items()}
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        example = example_proto.SerializeToString()
        examples.append(example)
    return examples, n_samples


def viz(ax, points, color, label=None):
    # turn actions sequences into 'trajectories' something we can plot
    for points_i in points:
        points_i = points_i.numpy()
        ax.plot(points_i[:, 0], points_i[:, 1], c=color, alpha=0.4, linewidth=3, label=label)
        ax.scatter(points_i[0, 0], points_i[0, 1], c=color, alpha=0.4, s=50)


def main_generate(args):
    root = pathlib.Path("recovery_data") / args.name
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'hparams.json').open("w") as f:
        dataset_hparams = {
            "state_keys": [
                "link_bot"
            ],
            "fwd_model_hparams": {
                "dynamics_dataset_hparams": {
                    "n_action": 2
                }
            }
        }
        json.dump(dataset_hparams, f)
    for mode, n_records in [('train', 10), ('test', 1), ('val', 5)]:
        full_output_directory = root / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)
        for record_idx in range(n_records):
            examples, n_samples = generate_examples()

            serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

            start_example_idx = record_idx * n_samples
            end_example_idx = start_example_idx + n_samples
            record_filename = "example_{:09d}_to_{:09d}.tfrecords".format(start_example_idx, end_example_idx - 1)
            full_filename = full_output_directory / record_filename
            print(full_filename)
            writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type='ZLIB')
            writer.write(serialized_dataset)


def main_sample_and_viz(args):
    n_samples = 100
    n_actions = 5
    starting_points = sample_starting_points(n_samples)
    actions, points = sample_actions_from_true_distribution(n_samples, n_actions, starting_points)
    plt.figure()
    ax = plt.gca()
    plt.axis("equal")
    viz(ax, points, 'b')
    plt.show()


def main_sample(args):
    recovery_actions_model = RNNRecoveryModelWrapper(args.checkpoint, get_scenario('link_bot'))

    dataset_dirs = recovery_actions_model.model_hparams['datasets']
    test_dataset = RecoveryDataset(paths_from_json(dataset_dirs))
    tf_dataset = test_dataset.get_datasets(mode='test', take=args.take)
    sampled_actions = []
    starting_points = []
    gt_actions = []
    for example in tf_dataset:
        environment = {
            'full_env/env': example['full_env/env'],
            'full_env/extent': example['full_env/extent'],
            'full_env/res': example['full_env/res'],
            'full_env/origin': example['full_env/origin'],
        }
        state = {k: example[add_predicted(k)][0] for k in test_dataset.state_keys}

        actions_i = recovery_actions_model.sample(environment, state)
        sampled_actions.append(actions_i)
        starting_points.append(state['link_bot'])
        gt_actions.append(example['action'])

    sampled_actions = tf.stack(sampled_actions, axis=0)
    starting_points = tf.stack(starting_points, axis=0)
    sampled_points = make_points(starting_points, sampled_actions)
    gt_points = make_points(starting_points, gt_actions)

    plt.figure()
    ax = plt.gca()
    plt.axis("equal")
    viz(ax, gt_points, color='b', label='true samples')
    viz(ax, sampled_points, color='r', label='predicted samples')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def main_viz_dataset(args):
    test_dataset = RecoveryDataset(args.dataset_dirs)
    tf_dataset = test_dataset.get_datasets(mode='test', take=args.take)
    starting_points = []
    gt_actions = []
    for example in tf_dataset:
        starting_points.append(example[add_predicted('link_bot')][0])
        gt_actions.append(example['action'])

    starting_points = tf.stack(starting_points, axis=0)
    gt_points = make_points(starting_points, gt_actions)

    plt.figure()
    ax = plt.gca()
    plt.axis("equal")
    viz(ax, gt_points, color='b')
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3, linewidth=250)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate_parser = subparsers.add_parser('generate')
    generate_parser.set_defaults(func=main_generate)
    generate_parser.add_argument('name')

    sample_and_viz_parser = subparsers.add_parser('viz')
    sample_and_viz_parser.set_defaults(func=main_sample_and_viz)

    viz_dataset_parser = subparsers.add_parser('viz_dataset')
    viz_dataset_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    viz_dataset_parser.add_argument('--take', type=int)
    viz_dataset_parser.set_defaults(func=main_viz_dataset)

    sample_parser = subparsers.add_parser('sample')
    sample_parser.set_defaults(func=main_sample)
    sample_parser.add_argument('checkpoint', type=pathlib.Path)
    sample_parser.add_argument('--take', type=int)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)
