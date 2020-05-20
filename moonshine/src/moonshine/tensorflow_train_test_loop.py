import json
import pathlib
from copy import deepcopy
from typing import Dict, Callable, Optional, List, Type

import numpy as np
import progressbar
import tensorflow as tf
from colorama import Fore, Style

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine import experiments_util
from moonshine.metric import Metric, LossMetric


class MyKerasModel(tf.keras.Model):
    """
    the "call" method is expected to take and return a dictionary
    """

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.batch_size = batch_size
        self.scenario = scenario


def compute_loss_and_metrics(tf_dataset, keras_model, loss_function, metrics_function, postprocess):
    losses = []
    metrics = {}
    for dataset_element in progressbar.progressbar(tf_dataset):
        if postprocess is not None:
            dataset_element = postprocess(dataset_element)
        predictions = keras_model(dataset_element, training=False)
        batch_loss = loss_function(dataset_element, predictions)
        losses.append(batch_loss)

        metrics_element = metrics_function(dataset_element, predictions)
        for k, v in metrics_element.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    mean_loss = np.mean(losses)

    mean_metrics = {}
    for k, v in metrics.items():
        mean_metrics[k] = np.mean(v)
    return mean_loss, mean_metrics


def train(keras_model: MyKerasModel,
          model_hparams: Dict,
          train_tf_dataset,
          val_tf_dataset,
          dataset_dirs: List[pathlib.Path],
          seed: int,
          batch_size: int,
          epochs: int,
          loss_function: Callable,
          metrics_function: Callable,
          postprocess: Optional[Callable] = None,
          checkpoint: Optional[pathlib.Path] = None,
          log_path: Optional[pathlib.Path] = None,
          log_scalars_every: int = 500,
          validation_every: int = 1,
          key_metric: Type[Metric] = LossMetric,
          ):
    """

    :param keras_model: the instantiated model
    :param model_hparams:
    :param train_tf_dataset:
    :param val_tf_dataset:
    :param dataset_dirs:
    :param seed:
    :param batch_size:
    :param epochs:
    :param loss_function: Takes an element of the dataset and the predictions on that element and returns the loss
    :param metrics_function: Takes an element of the dataset and the predictions on that element and returns a dict of metrics
    :param postprocess: function
    :param checkpoint:
    :param log_path:
    :param log_scalars_every:
    :param validation_every:
    :param key_metric: Used to determine what the "best" model is for saving
    :param ensemble: number of times to copy the model
    :return:
    """
    if 'learning_rate_schedule' in model_hparams:
        print("Using learning rate schedule")
        learning_rate_schedule_params = model_hparams['learning_rate_schedule']
        initial_learning_rate = learning_rate_schedule_params['initial_rate']
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                       decay_steps=learning_rate_schedule_params['decay_steps'],
                                                                       decay_rate=learning_rate_schedule_params['decay_rate'])
    else:
        learning_rate = model_hparams['learning_rate']
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    global_step = tf.Variable(1)

    # If we're resuming a checkpoint, there is no new log path
    if checkpoint is not None:
        full_log_path = checkpoint
        logging = True
    elif log_path:
        full_log_path = pathlib.Path("log_data") / log_path
        logging = True
    else:
        full_log_path = '/tmp'
        logging = False

    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, net=keras_model)
    manager = tf.train.CheckpointManager(ckpt, full_log_path, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if checkpoint is not None:
        if manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
        else:
            print(Fore.RED + "Failed to restore from checkpoint directory {}".format(checkpoint) + Fore.RESET)
            print("Did you forget a subdirectory?")
            return

    writer = None
    if logging:
        print(Fore.CYAN + "Logging to {}".format(full_log_path) + Fore.RESET)

        experiments_util.make_log_dir(full_log_path)

        hparams_path = full_log_path / "hparams.json"
        with hparams_path.open('w') as hparams_file:
            model_hparams['log path'] = str(full_log_path)
            model_hparams['seed'] = seed
            model_hparams['batch_size'] = batch_size
            model_hparams['dataset'] = [str(dataset_dir) for dataset_dir in dataset_dirs]
            model_hparams['key_metric'] = key_metric.key()
            hparams_file.write(json.dumps(model_hparams, indent=2))

        writer = tf.summary.create_file_writer(logdir=str(full_log_path))

    @tf.function
    def forward_pass_and_apply_gradients(train_element):
        with tf.GradientTape() as tape:
            train_predictions = keras_model(train_element, training=True)
            train_batch_loss = loss_function(train_element, train_predictions)

            flooding_level = model_hparams['flooding_level'] if 'flooding_level' in model_hparams else None
            if flooding_level is not None:
                train_batch_loss = tf.math.abs(train_batch_loss - flooding_level) + flooding_level

        variables = keras_model.trainable_variables
        gradients = tape.gradient(train_batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return train_predictions, train_batch_loss

    def train_loop():
        step = None
        best_key_metric_value = None
        validation_iterator = iter(val_tf_dataset.repeat())

        for epoch in range(epochs):
            ################
            # train
            ################
            # metrics are averaged across batches in the epoch
            batch_losses = []

            for train_element in progressbar.progressbar(train_tf_dataset):
                step = int(global_step.numpy())
                if postprocess is not None:
                    train_element = postprocess(train_element)

                train_predictions, train_batch_loss = forward_pass_and_apply_gradients(train_element)
                batch_losses.append(train_batch_loss.numpy())

                if logging:
                    if step % log_scalars_every == 0:
                        tf.summary.scalar("batch_loss", train_batch_loss, step=step)

                        train_batch_metrics = metrics_function(train_element, train_predictions)
                        for metric_name, metric_value in train_batch_metrics.items():
                            tf.summary.scalar('train_' + metric_name.replace(" ", "_"), metric_value, step=step)

                    if step % 500 == 0:
                        val_losses = []
                        val_metrics = {}
                        for i in range(32):
                            val_dataset_element = next(validation_iterator)
                            if postprocess is not None:
                                val_dataset_element = postprocess(val_dataset_element)
                            predictions = keras_model(val_dataset_element, training=False)
                            val_batch_loss = loss_function(val_dataset_element, predictions)
                            val_losses.append(val_batch_loss)

                            metrics_element = metrics_function(val_dataset_element, predictions)
                            for k, v in metrics_element.items():
                                if k not in val_metrics:
                                    val_metrics[k] = []
                                val_metrics[k].append(v)

                        val_mean_loss = np.mean(val_losses)

                        val_mean_metrics = {}
                        for k, v in val_metrics.items():
                            val_mean_metrics[k] = np.mean(v)
                        if logging:
                            tf.summary.scalar('validation_loss', val_mean_loss, step=step)
                            for metric_name, mean_metric_value in val_mean_metrics.items():
                                tf.summary.scalar('validation_' + metric_name.replace(" ", "_"), mean_metric_value, step=step)

                ####################
                # Update global step
                ####################
                global_step.assign_add(1)

            training_loss = np.mean(batch_losses)
            log_msg = "Epoch: {:5d}, Training Loss: {:8.5f}"
            print(log_msg.format(epoch, training_loss))

            ################
            # validation
            ################
            if epoch % validation_every == 0:
                val_mean_loss, val_mean_metrics = compute_loss_and_metrics(val_tf_dataset,
                                                                           keras_model,
                                                                           loss_function,
                                                                           metrics_function,
                                                                           postprocess)

                log_msg = "Epoch: {:5d}, Validation Loss: {:8.5f}"
                print(Style.BRIGHT + log_msg.format(epoch, val_mean_loss) + Style.NORMAL)

                if logging:
                    tf.summary.scalar('validation_loss', val_mean_loss, step=step)
                    for metric_name, mean_metric_value in val_mean_metrics.items():
                        print(metric_name, mean_metric_value)
                        tf.summary.scalar('validation_' + metric_name.replace(" ", "_"), mean_metric_value, step=step)

                # check new best based on the desired metric (or loss)
                if key_metric.key() == 'loss':
                    key_metric_value = val_mean_loss
                else:
                    key_metric_value = val_mean_metrics[key_metric.key()]

                if best_key_metric_value is None or key_metric.is_better_than(key_metric_value, best_key_metric_value):
                    best_key_metric_value = key_metric_value
                    if logging:
                        save_path = manager.save()
                        print(Fore.CYAN + "Step {:6d}: Saved checkpoint {}".format(step, save_path) + Fore.RESET)

        if not logging:
            # save the last model, which will be saved in tmp, just in case we did want it
            save_path = manager.save()
            print(Fore.CYAN + "Step {:6d}: Saved final checkpoint {}".format(step, save_path) + Fore.RESET)

    try:
        if logging:
            with writer.as_default():
                train_loop()
        else:
            train_loop()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)


def evaluate(keras_model: MyKerasModel,
             test_tf_dataset,
             loss_function: Callable,
             checkpoint_path: pathlib.Path,
             metrics_function: Callable,
             postprocess: Optional[Callable] = None,
             ):
    ckpt = tf.train.Checkpoint(net=keras_model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)  # doesn't matter here, we're not saving
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)

    try:
        test_mean_loss, test_mean_metrics = compute_loss_and_metrics(test_tf_dataset,
                                                                     keras_model,
                                                                     loss_function,
                                                                     metrics_function,
                                                                     postprocess)
        print("Test Loss:  {:8.5f}".format(test_mean_loss))

        for metric_name, metric_value in test_mean_metrics.items():
            print("{} {:8.4f}".format(metric_name, metric_value))
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
