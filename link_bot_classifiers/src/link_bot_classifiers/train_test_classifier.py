#!/usr/bin/env python
import json
import pathlib
import time
from typing import List, Optional

import numpy as np
import tensorflow as tf

import link_bot_classifiers
import rospy
from link_bot_classifiers.classifier_utils import load_generic_model
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import add_predicted, batch_tf_dataset, balance
from link_bot_pycommon.collision_checking import batch_in_collision_tf_3d
from link_bot_pycommon.pycommon import paths_to_json
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import (index_dict_of_batched_vectors_tf,
                                       sequence_of_dicts_to_dict_of_sequences)
from shape_completion_training.metric import AccuracyMetric
from shape_completion_training.model import filepath_tools
from shape_completion_training.model.utils import reduce_mean_dict
from shape_completion_training.model_runner import ModelRunner
from std_msgs.msg import Float32


def train_main(dataset_dirs: List[pathlib.Path],
               model_hparams: pathlib.Path,
               log: str,
               batch_size: int,
               epochs: int,
               seed: int,
               checkpoint: Optional[pathlib.Path] = None,
               take: Optional[int] = None,
               ensemble_idx: Optional[int] = None,
               trials_directory: Optional[pathlib.Path] = None,
               **kwargs):
    ###############
    # Datasets
    ###############
    # set load_true_states=True when debugging
    train_dataset = ClassifierDataset(dataset_dirs, load_true_states=True)
    val_dataset = ClassifierDataset(dataset_dirs, load_true_states=True)

    ###############
    # Model
    ###############
    model_hparams = json.load((model_hparams).open('r'))
    model_hparams['classifier_dataset_hparams'] = train_dataset.hparams
    model_hparams['batch_size'] = batch_size
    model_hparams['seed'] = seed
    model_hparams['latest_training_time'] = int(time.time())
    model_hparams['datasets'] = paths_to_json(dataset_dirs)
    trial_path = None
    if checkpoint:
        trial_path = checkpoint.parent.absolute()
    group_name = log if trial_path is None else None
    if ensemble_idx is not None:
        group_name = f"{group_name}_{ensemble_idx}"
    trial_path, _ = filepath_tools.create_or_load_trial(group_name=group_name,
                                                        params=model_hparams,
                                                        trial_path=trial_path,
                                                        trials_directory=trials_directory,
                                                        write_summary=False)
    model_class = link_bot_classifiers.get_model(model_hparams['model_class'])

    model = model_class(hparams=model_hparams, batch_size=batch_size, scenario=train_dataset.scenario)

    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         trial_path=trial_path,
                         key_metric=AccuracyMetric,
                         checkpoint=checkpoint,
                         mid_epoch_val_batches=100,
                         val_every_n_batches=1000,
                         save_every_n_minutes=20,
                         validate_first=True,
                         batch_metadata=train_dataset.batch_metadata)

    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train', take=take, shuffle_files=True)
    val_tf_dataset = val_dataset.get_datasets(mode='val', take=take, shuffle_files=True)

    train_tf_dataset = train_tf_dataset.shuffle(256, reshuffle_each_iteration=True)

    rospy.logerr_once("NOT BALANCING!")
    # train_tf_dataset = balance(train_tf_dataset)
    # val_tf_dataset = balance(val_tf_dataset)

    train_tf_dataset = batch_tf_dataset(train_tf_dataset, batch_size, drop_remainder=True)
    val_tf_dataset = batch_tf_dataset(val_tf_dataset, batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path


def test_main(dataset_dirs: List[pathlib.Path],
              take: int,
              mode: str,
              batch_size: int,
              checkpoint: Optional[pathlib.Path] = None,
              trials_directory=pathlib.Path,
              **kwargs):
    ###############
    # Model
    ###############
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)
    model = link_bot_classifiers.get_model(params['model_class'])

    ###############
    # Dataset
    ###############
    test_dataset = ClassifierDataset(dataset_dirs, load_true_states=True)
    test_tf_dataset = test_dataset.get_datasets(mode=mode, take=take)
    test_tf_dataset = balance(test_tf_dataset)
    scenario = test_dataset.scenario

    ###############
    # Evaluate
    ###############
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, batch_size, drop_remainder=True)

    net = model(hparams=params, batch_size=batch_size, scenario=test_dataset.scenario)
    # This call to model runner restores the model
    runner = ModelRunner(model=net,
                         training=False,
                         params=params,
                         checkpoint=checkpoint,
                         trial_path=trial_path,
                         key_metric=AccuracyMetric,
                         batch_metadata=test_dataset.batch_metadata)

    metrics = runner.val_epoch(test_tf_dataset)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:30s}: {metric_value}")


def eval_main(dataset_dirs: List[pathlib.Path],
              checkpoint: pathlib.Path,
              mode: str,
              batch_size: int,
              only_errors: bool,
              **kwargs):
    stdev_pub_ = rospy.Publisher("stdev", Float32, queue_size=10)
    accept_probability_pub_ = rospy.Publisher("accept_probability_viz", Float32, queue_size=10)
    traj_idx_pub_ = rospy.Publisher("traj_idx_viz", Float32, queue_size=10)

    ###############
    # Model
    ###############
    trials_directory = pathlib.Path('trials').absolute()
    trial_path = checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)
    model = link_bot_classifiers.get_model(params['model_class'])

    ###############
    # Dataset
    ###############
    test_dataset = ClassifierDataset(dataset_dirs, load_true_states=True)
    test_tf_dataset = test_dataset.get_datasets(mode=mode)
    scenario = test_dataset.scenario

    ###############
    # Evaluate
    ###############
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, batch_size, drop_remainder=True)

    net = model(hparams=params, batch_size=batch_size, scenario=test_dataset.scenario)
    # This call to model runner restores the model
    runner = ModelRunner(model=net,
                         training=False,
                         params=params,
                         checkpoint=checkpoint,
                         trial_path=trial_path,
                         key_metric=AccuracyMetric,
                         batch_metadata=test_dataset.batch_metadata)

    # Iterate over test set
    all_accuracies_over_time = []
    test_metrics = []
    all_stdevs = []
    all_labels = []
    for batch_idx, test_batch in enumerate(test_tf_dataset):
        print(batch_idx)
        test_batch.update(test_dataset.batch_metadata)

        predictions, test_batch_metrics = runner.model.val_step(test_batch)

        test_metrics.append(test_batch_metrics)
        labels = tf.expand_dims(test_batch['is_close'][:, 1:], axis=2)

        all_labels = tf.concat((all_labels, tf.reshape(test_batch['is_close'][:, 1:], [-1])), axis=0)
        all_stdevs = tf.concat((all_stdevs, tf.reshape(test_batch[add_predicted('stdev')], [-1])), axis=0)

        probabilities = predictions['probabilities']
        accuracy_over_time = tf.keras.metrics.binary_accuracy(y_true=labels, y_pred=probabilities)
        all_accuracies_over_time.append(accuracy_over_time)

        # Visualization
        test_batch.pop("time")
        test_batch.pop("batch_size")
        decisions = probabilities > 0.5
        classifier_is_correct = tf.squeeze(tf.equal(decisions, tf.cast(labels, tf.bool)), axis=-1)
        for b in range(batch_size):
            example = index_dict_of_batched_vectors_tf(test_batch, b)

            # if the classifier is correct at all time steps, ignore
            if only_errors and tf.reduce_all(classifier_is_correct[b]):
                continue

            # if only_collision
            predicted_rope_states = tf.reshape(example[add_predicted('link_bot')][1], [-1, 3])
            xs = predicted_rope_states[:, 0]
            ys = predicted_rope_states[:, 1]
            zs = predicted_rope_states[:, 2]
            in_collision = bool(batch_in_collision_tf_3d(environment=example,
                                                         xs=xs, ys=ys, zs=zs,
                                                         inflate_radius_m=0)[0].numpy())
            label = bool(example['is_close'][1].numpy())
            accept = decisions[b, 0, 0].numpy()
            if not (in_collision and accept):
                continue

            time_steps = np.arange(test_dataset.horizon)
            scenario.plot_environment_rviz(example)
            anim = RvizAnimationController(time_steps)
            while not anim.done:
                t = anim.t()
                # use scenario plot transition function here
                scenario.plot_transition_rviz(example, t)

                # TODO: reconsider where this goes, see visualize_classifier_dataset.py
                stdev_t = example[add_predicted('stdev')][t, 0].numpy()
                stdev_msg = Float32()
                stdev_msg.data = stdev_t
                stdev_pub_.publish(stdev_msg)

                if t > 0:
                    accept_probability_t = predictions['probabilities'][b, t - 1, 0].numpy()
                else:
                    accept_probability_t = -999
                accept_probability_msg = Float32()
                accept_probability_msg.data = accept_probability_t
                accept_probability_pub_.publish(accept_probability_msg)

                traj_idx_msg = Float32()
                traj_idx_msg.data = batch_idx * batch_size + b
                traj_idx_pub_.publish(traj_idx_msg)

                # this will return when either the animation is "playing" or because the user stepped forward
                anim.step()

    all_accuracies_over_time = tf.concat(all_accuracies_over_time, axis=0)
    mean_accuracies_over_time = tf.reduce_mean(all_accuracies_over_time, axis=0)
    std_accuracies_over_time = tf.math.reduce_std(all_accuracies_over_time, axis=0)

    test_metrics = sequence_of_dicts_to_dict_of_sequences(test_metrics)
    mean_test_metrics = reduce_mean_dict(test_metrics)
    for metric_name, metric_value in mean_test_metrics.items():
        metric_value_str = np.format_float_positional(metric_value, precision=4, unique=False, fractional=False)
        print(f"{metric_name}: {metric_value_str}")

    import matplotlib.pyplot as plt
    plt.style.use("slides")
    time_steps = np.arange(1, test_dataset.horizon)
    plt.plot(time_steps, mean_accuracies_over_time, label='mean', color='r')
    plt.plot(time_steps, mean_accuracies_over_time - std_accuracies_over_time, color='orange', alpha=0.5)
    plt.plot(time_steps, mean_accuracies_over_time + std_accuracies_over_time, color='orange', alpha=0.5)
    plt.fill_between(time_steps,
                     mean_accuracies_over_time - std_accuracies_over_time,
                     mean_accuracies_over_time + std_accuracies_over_time,
                     label="68% confidence interval",
                     color='r',
                     alpha=0.3)
    plt.ylim(0, 1.05)
    plt.title("classifier accuracy versus horizon")
    plt.xlabel("time step")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def eval_ensemble_main(dataset_dir: pathlib.Path,
                       checkpoints: List[pathlib.Path],
                       mode: str,
                       batch_size: int,
                       only_errors: bool,
                       **kwargs):
    dynamics_stdev_pub_ = rospy.Publisher("dynamics_stdev", Float32, queue_size=10)
    classifier_stdev_pub_ = rospy.Publisher("classifier_stdev", Float32, queue_size=10)
    accept_probability_pub_ = rospy.Publisher("accept_probability_viz", Float32, queue_size=10)
    traj_idx_pub_ = rospy.Publisher("traj_idx_viz", Float32, queue_size=10)

    ###############
    # Model
    ###############
    model = load_generic_model(checkpoints)

    ###############
    # Dataset
    ###############
    test_dataset = ClassifierDataset([dataset_dir], load_true_states=True)
    test_tf_dataset = test_dataset.get_datasets(mode=mode)
    test_tf_dataset = batch_tf_dataset(test_tf_dataset, batch_size, drop_remainder=True)
    scenario = test_dataset.scenario

    ###############
    # Evaluate
    ###############

    # Iterate over test set
    all_accuracies_over_time = []
    all_stdevs = []
    all_labels = []
    classifier_ensemble_stdevs = []
    for batch_idx, test_batch in enumerate(test_tf_dataset):
        test_batch.update(test_dataset.batch_metadata)

        mean_predictions, stdev_predictions = model.check_constraint_from_example(test_batch)
        mean_probabilities = mean_predictions['probabilities']
        stdev_probabilities = stdev_predictions['probabilities']

        labels = tf.expand_dims(test_batch['is_close'][:, 1:], axis=2)

        all_labels = tf.concat((all_labels, tf.reshape(test_batch['is_close'][:, 1:], [-1])), axis=0)
        all_stdevs = tf.concat((all_stdevs, tf.reshape(test_batch[add_predicted('stdev')], [-1])), axis=0)

        accuracy_over_time = tf.keras.metrics.binary_accuracy(y_true=labels, y_pred=mean_probabilities)
        all_accuracies_over_time.append(accuracy_over_time)

        # Visualization
        test_batch.pop("time")
        test_batch.pop("batch_size")
        decisions = mean_probabilities > 0.5
        classifier_is_correct = tf.squeeze(tf.equal(decisions, tf.cast(labels, tf.bool)), axis=-1)
        for b in range(batch_size):
            example = index_dict_of_batched_vectors_tf(test_batch, b)

            classifier_ensemble_stdev = stdev_probabilities[b].numpy().squeeze()
            classifier_ensemble_stdevs.append(classifier_ensemble_stdev)

            # if the classifier is correct at all time steps, ignore
            if only_errors and tf.reduce_all(classifier_is_correct[b]):
                continue

            # if only_collision
            predicted_rope_states = tf.reshape(example[add_predicted('link_bot')][1], [-1, 3])
            xs = predicted_rope_states[:, 0]
            ys = predicted_rope_states[:, 1]
            zs = predicted_rope_states[:, 2]
            in_collision = bool(batch_in_collision_tf_3d(environment=example,
                                                         xs=xs, ys=ys, zs=zs,
                                                         inflate_radius_m=0)[0].numpy())
            label = bool(example['is_close'][1].numpy())
            accept = decisions[b, 0, 0].numpy()
            # if not (in_collision and accept):
            #     continue

            scenario.plot_environment_rviz(example)

            stdev_probabilities[b].numpy().squeeze()
            classifier_stdev_msg = Float32()
            classifier_stdev_msg.data = stdev_probabilities[b].numpy().squeeze()
            classifier_stdev_pub_.publish(classifier_stdev_msg)

            actual_0 = scenario.index_state_time(example, 0)
            actual_1 = scenario.index_state_time(example, 1)
            pred_0 = scenario.index_predicted_state_time(example, 0)
            pred_1 = scenario.index_predicted_state_time(example, 1)
            action = scenario.index_action_time(example, 0)
            label = example['is_close'][1]
            scenario.plot_state_rviz(actual_0, label='actual', color='#FF0000AA', idx=0)
            scenario.plot_state_rviz(actual_1, label='actual', color='#E00016AA', idx=1)
            scenario.plot_state_rviz(pred_0, label='predicted', color='#0000FFAA', idx=0)
            scenario.plot_state_rviz(pred_1, label='predicted', color='#0553FAAA', idx=1)
            scenario.plot_action_rviz(pred_0, action)
            scenario.plot_is_close(label)

            dynamics_stdev_t = example[add_predicted('stdev')][1, 0].numpy()
            dynamics_stdev_msg = Float32()
            dynamics_stdev_msg.data = dynamics_stdev_t
            dynamics_stdev_pub_.publish(dynamics_stdev_msg)

            accept_probability_t = mean_probabilities[b, 0, 0].numpy()
            accept_probability_msg = Float32()
            accept_probability_msg.data = accept_probability_t
            accept_probability_pub_.publish(accept_probability_msg)

            traj_idx_msg = Float32()
            traj_idx_msg.data = batch_idx * batch_size + b
            traj_idx_pub_.publish(traj_idx_msg)

            # stepper = RvizSimpleStepper()
            # stepper.step()

        print(np.mean(classifier_ensemble_stdevs))

    all_accuracies_over_time = tf.concat(all_accuracies_over_time, axis=0)
    mean_accuracies_over_time = tf.reduce_mean(all_accuracies_over_time, axis=0)
    std_accuracies_over_time = tf.math.reduce_std(all_accuracies_over_time, axis=0)
    mean_classifier_ensemble_stdev = tf.reduce_mean(classifier_ensemble_stdevs)
    print(mean_classifier_ensemble_stdev)
