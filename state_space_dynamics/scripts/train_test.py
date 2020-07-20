#!/usr/bin/env python
import argparse
import json
import time
import pathlib

import numpy as np
import tensorflow as tf

import rospy
import state_space_dynamics
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import add_predicted, batch_tf_dataset
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from link_bot_pycommon.pycommon import paths_to_json
from moonshine.moonshine_utils import add_batch, remove_batch
from moonshine.gpu_config import limit_gpu_mem
from shape_completion_training.model import filepath_tools
from shape_completion_training.model_runner import ModelRunner

limit_gpu_mem(8)


def train_main(args, seed: int):
    ###############
    # Datasets
    ###############
    train_dataset = DynamicsDataset(args.dataset_dirs)
    val_dataset = DynamicsDataset(args.dataset_dirs)

    ###############
    # Model
    ###############
    model_hparams = json.load((args.model_hparams).open('r'))
    model_hparams['dynamics_dataset_hparams'] = train_dataset.hparams
    model_hparams['batch_size'] = args.batch_size
    model_hparams['seed'] = seed
    model_hparams['datasets'] = paths_to_json(args.dataset_dirs)
    model_hparams['latest_training_time'] = int(time.time())
    trial_path = None
    checkpoint_name = None
    if args.checkpoint:
        trial_path = args.checkpoint.parent.absolute()
        checkpoint_name = args.checkpoint.name
    trials_directory = pathlib.Path('trials')
    group_name = args.log if trial_path is None else None
    if args.ensemble_idx is not None:
        group_name = f"{group_name}_{args.ensemble_idx}"
    trial_path, _ = filepath_tools.create_or_load_trial(group_name=group_name,
                                                        params=model_hparams,
                                                        trial_path=trial_path,
                                                        trials_directory=trials_directory,
                                                        write_summary=False)
    model_class = state_space_dynamics.get_model(model_hparams['model_class'])

    model = model_class(hparams=model_hparams, batch_size=args.batch_size, scenario=train_dataset.scenario)
    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         restore_from_name=checkpoint_name,
                         batch_metadata=train_dataset.batch_metadata,
                         trial_path=trial_path)

    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train')
    val_tf_dataset = val_dataset.get_datasets(mode='val', take=args.take)

    # to mix up examples so each batch is diverse
    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=512, seed=seed, reshuffle_each_iteration=True)

    train_tf_dataset = batch_tf_dataset(train_tf_dataset, args.batch_size, drop_remainder=True)
    val_tf_dataset = batch_tf_dataset(val_tf_dataset, args.batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.shuffle(
        buffer_size=128, seed=seed, reshuffle_each_iteration=True)  # to mix up batches

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=args.epochs)


def eval_main(args, seed: int):
    test_dataset = DynamicsDataset(args.dataset_dirs)

    trials_directory = pathlib.Path('trials').absolute()
    trial_path = args.checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)
    model = state_space_dynamics.get_model(params['model_class'])
    net = model(hparams=params, batch_size=args.batch_size, scenario=test_dataset.scenario)

    runner = ModelRunner(model=net,
                         training=False,
                         restore_from_name=args.checkpoint.name,
                         batch_metadata=test_dataset.batch_metadata,
                         trial_path=trial_path,
                         params=params)

    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, args.batch_size, drop_remainder=True)
    validation_metrics = runner.val_epoch(test_tf_dataset)
    for name, value in validation_metrics.items():
        print(f"{name}: {value}")

    # more metrics that can't be expressed as just an average over metrics on each batch
    all_errors = None
    for batch in test_tf_dataset:
        outputs = runner.model(batch, training=False)
        zs = tf.reshape(batch['link_bot'], [-1, 3])[:, 2]
        print(tf.reduce_min(zs))
        print(tf.reduce_max(zs))
        errors_for_batch = tf.linalg.norm(outputs['link_bot'] - batch['link_bot'], axis=2)
        if all_errors is not None:
            all_errors = tf.concat([all_errors, errors_for_batch], axis=0)
        else:
            all_errors = errors_for_batch
    print(f"90th percentile {np.percentile(all_errors.numpy(), 90)}")
    print(f"95th percentile {np.percentile(all_errors.numpy(), 95)}")
    print(f"99th percentile {np.percentile(all_errors.numpy(), 99)}")
    print(f"max {np.max(all_errors.numpy())}")


def viz_main(args, seed: int):
    test_dataset = DynamicsDataset(args.dataset_dirs)

    trials_directory = pathlib.Path('trials').absolute()
    trial_path = args.checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)
    model = state_space_dynamics.get_model(params['model_class'])
    net = model(hparams=params, batch_size=1, scenario=test_dataset.scenario)

    runner = ModelRunner(model=net,
                         training=False,
                         restore_from_name=args.checkpoint.name,
                         batch_metadata=test_dataset.batch_metadata,
                         trial_path=trial_path,
                         params=params)

    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, 1, drop_remainder=True)

    for i, batch in enumerate(test_tf_dataset):
        if i < 100:
            continue
        print(i)
        batch.update(test_dataset.batch_metadata)
        predictions = runner.model(batch, training=False)

        test_dataset.scenario.plot_environment_rviz(remove_batch(batch))
        anim = RvizAnimationController(np.arange(test_dataset.sequence_length))
        while not anim.done:
            t = anim.t()
            actual_t = remove_batch(test_dataset.scenario.index_state_time(batch, t))
            action_t = remove_batch(test_dataset.scenario.index_action_time(batch, t))
            test_dataset.scenario.plot_state_rviz(actual_t, label='actual', color='red')
            test_dataset.scenario.plot_action_rviz(actual_t, action_t, color='gray')
            prediction_t = remove_batch(test_dataset.scenario.index_state_time(predictions, t))
            test_dataset.scenario.plot_state_rviz(prediction_t, label='predicted', color='blue')

            anim.step()


def main():
    np.set_printoptions(linewidth=250, precision=4, suppress=True, threshold=10000)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=16)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--epochs', type=int, default=500)
    train_parser.add_argument('--ensemble-idx', type=int)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches',
                              default=100)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs',
                              default=1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=16)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--seed', type=int, default=None)
    eval_parser.set_defaults(func=eval_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    viz_parser.add_argument('checkpoint', type=pathlib.Path)
    viz_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    viz_parser.add_argument('--verbose', '-v', action='count', default=0)
    viz_parser.add_argument('--seed', type=int, default=None)
    viz_parser.set_defaults(func=viz_main)

    args = parser.parse_args()

    from time import time
    now = str(int(time()))
    name = f"train_test_{now}"
    rospy.init_node(name)

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Using seed {}".format(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
