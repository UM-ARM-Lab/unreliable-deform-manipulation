#!/usr/bin/env python
import json
import pathlib
import time
from typing import List, Optional

import numpy as np
import tensorflow as tf

import state_space_dynamics
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import batch_tf_dataset
from link_bot_pycommon.pycommon import paths_to_json
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import remove_batch
from shape_completion_training.model import filepath_tools
from shape_completion_training.model_runner import ModelRunner


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               checkpoint: Optional[pathlib.Path] = None,
               ensemble_idx: Optional[int] = None,
               trials_directory=pathlib.Path,
               ):
    ###############
    # Datasets
    ###############
    train_dataset = DynamicsDataset(dataset_dirs)
    val_dataset = DynamicsDataset(dataset_dirs)

    ###############
    # Model
    ###############
    model_hparams = json.load((model_hparams).open('r'))
    model_hparams['dynamics_dataset_hparams'] = train_dataset.hparams
    model_hparams['batch_size'] = batch_size
    model_hparams['seed'] = seed
    model_hparams['datasets'] = paths_to_json(dataset_dirs)
    model_hparams['latest_training_time'] = int(time.time())
    trial_path = None
    checkpoint_name = None
    if checkpoint:
        trial_path = checkpoint.parent.absolute()
        checkpoint_name = checkpoint.name
    group_name = log if trial_path is None else None
    if ensemble_idx is not None:
        group_name = f"{group_name}_{ensemble_idx}"
    trial_path, _ = filepath_tools.create_or_load_trial(group_name=group_name,
                                                        params=model_hparams,
                                                        trial_path=trial_path,
                                                        trials_directory=trials_directory,
                                                        write_summary=False)
    model_class = state_space_dynamics.get_model(model_hparams['model_class'])

    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.scenario)
    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         restore_from_name=checkpoint_name,
                         batch_metadata=train_dataset.batch_metadata,
                         trial_path=trial_path)

    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train')
    val_tf_dataset = val_dataset.get_datasets(mode='val')

    # to mix up examples so each batch is diverse
    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=512, seed=seed, reshuffle_each_iteration=True)

    train_tf_dataset = batch_tf_dataset(train_tf_dataset, batch_size, drop_remainder=True)
    val_tf_dataset = batch_tf_dataset(val_tf_dataset, batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.shuffle(
        buffer_size=128, seed=seed, reshuffle_each_iteration=True)  # to mix up batches

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path


def eval_main(dataset_dirs: List[pathlib.Path],
              checkpoint: pathlib.Path,
              mode: str,
              batch_size: int,
              ):
    test_dataset = DynamicsDataset(dataset_dirs)

    trials_directory = pathlib.Path('trials').absolute()
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)
    model = state_space_dynamics.get_model(params['model_class'])
    net = model(hparams=params, batch_size=batch_size, scenario=test_dataset.scenario)

    runner = ModelRunner(model=net,
                         training=False,
                         restore_from_name=checkpoint.name,
                         batch_metadata=test_dataset.batch_metadata,
                         trial_path=trial_path,
                         params=params)

    test_tf_dataset = test_dataset.get_datasets(mode=mode)
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, batch_size, drop_remainder=True)
    validation_metrics = runner.val_epoch(test_tf_dataset)
    for name, value in validation_metrics.items():
        print(f"{name}: {value}")

    # more metrics that can't be expressed as just an average over metrics on each batch
    all_errors = None
    for batch in test_tf_dataset:
        outputs = runner.model(batch, training=False)
        zs = tf.reshape(batch['link_bot'], [-1, 3])[:, 2]
        errors_for_batch = tf.linalg.norm(outputs['link_bot'] - batch['link_bot'], axis=2)
        if all_errors is not None:
            all_errors = tf.concat([all_errors, errors_for_batch], axis=0)
        else:
            all_errors = errors_for_batch
    print(f"90th percentile {np.percentile(all_errors.numpy(), 90)}")
    print(f"95th percentile {np.percentile(all_errors.numpy(), 95)}")
    print(f"99th percentile {np.percentile(all_errors.numpy(), 99)}")
    print(f"max {np.max(all_errors.numpy())}")


def viz_main(dataset_dirs: List[pathlib.Path],
             checkpoint: pathlib.Path,
             mode: str,
             ):
    test_dataset = DynamicsDataset(dataset_dirs)

    trials_directory = pathlib.Path('trials').absolute()
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)
    model = state_space_dynamics.get_model(params['model_class'])
    net = model(hparams=params, batch_size=1, scenario=test_dataset.scenario)

    runner = ModelRunner(model=net,
                         training=False,
                         restore_from_name=checkpoint.name,
                         batch_metadata=test_dataset.batch_metadata,
                         trial_path=trial_path,
                         params=params)

    test_tf_dataset = test_dataset.get_datasets(mode=mode)
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, 1, drop_remainder=True)

    for i, batch in enumerate(test_tf_dataset):
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
