import json
import pathlib
from typing import Dict, Callable, Optional, List

import numpy as np
import progressbar
import tensorflow as tf
from colorama import Fore, Style

from link_bot_planning.experiment_scenario import ExperimentScenario
from moonshine import experiments_util


class MyKerasModel(tf.keras.Model):
    """
    the "call" method is expected to take and return a dictionary
    """

    def __init__(self, hparams: Dict, batch_size: int, scenario: ExperimentScenario):
        super().__init__()
        self.hparams = tf.contrib.checkpoint.NoDependency(hparams)
        self.batch_size = batch_size
        self.scenario = scenario


def compute_loss_and_metrics(tf_dataset, net, loss_function, metrics_function):
    losses = []
    metrics = {}
    for dataset_element in progressbar.progressbar(tf_dataset):
        predictions = net(dataset_element)
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
          metrics_function: Optional[Callable],
          checkpoint: Optional[pathlib.Path] = None,
          log_path: Optional[pathlib.Path] = None,
          log_scalars_every: int = 500,
          validation_every: int = 1,
          key_metric: str = 'loss',
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
    :param checkpoint:
    :param log_path:
    :param log_scalars_every:
    :param validation_every:
    :param key_metric: Used to determine what the "best" model is for saving
    :param ensemble: number of times to copy the model
    :return:
    """
    optimizer = tf.train.AdamOptimizer()

    global_step = tf.train.get_or_create_global_step()

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
    ckpt.restore(manager.latest_checkpoint)
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
            hparams_file.write(json.dumps(model_hparams, indent=2))

        writer = tf.contrib.summary.create_file_writer(logdir=full_log_path)

    def train_loop():
        step = None
        best_key_metric_value = None

        for epoch in range(epochs):
            ################
            # train
            ################
            # metrics are averaged across batches in the epoch
            batch_losses = []

            for train_element in progressbar.progressbar(train_tf_dataset):
                step = int(global_step.numpy())

                with tf.GradientTape() as tape:
                    train_predictions = keras_model(train_element)
                    train_batch_loss = loss_function(train_element, train_predictions)

                variables = keras_model.trainable_variables
                gradients = tape.gradient(train_batch_loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                batch_losses.append(train_batch_loss.numpy())

                if logging:
                    if step % log_scalars_every == 0:
                        tf.contrib.summary.scalar("batch_loss", train_batch_loss, step=step)

                        if metrics_function:
                            train_batch_metrics = metrics_function(train_element, train_predictions)
                            for metric_name, metric_value in train_batch_metrics.items():
                                tf.contrib.summary.scalar('train_' + metric_name.replace(" ", "_"), metric_value, step=step)

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
                val_mean_loss, val_mean_metrics = compute_loss_and_metrics(val_tf_dataset, keras_model, loss_function,
                                                                           metrics_function)

                log_msg = "Epoch: {:5d}, Validation Loss: {:8.5f}"
                print(Style.BRIGHT + log_msg.format(epoch, val_mean_loss) + Style.NORMAL)

                if logging:
                    tf.contrib.summary.scalar('validation_loss', val_mean_loss, step=step)
                    for metric_name, mean_metric_value in val_mean_metrics.items():
                        print(metric_name, mean_metric_value)
                        tf.contrib.summary.scalar('validation_' + metric_name.replace(" ", "_"), mean_metric_value, step=step)

                # check new best based on the desired metric (or loss)
                if key_metric == 'loss':
                    key_metric_value = val_mean_loss
                else:
                    key_metric_value = val_mean_metrics[key_metric]

                if best_key_metric_value is None or key_metric_value < best_key_metric_value:
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
            with writer.as_default(), tf.contrib.summary.always_record_summaries():
                train_loop()
        else:
            train_loop()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)


def evaluate(keras_model: MyKerasModel,
             test_tf_dataset,
             loss_function: Callable,
             checkpoint_path: pathlib.Path,
             metrics_function: Optional[Callable],
             ):
    ckpt = tf.train.Checkpoint(net=keras_model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)  # doesn't matter here, we're not saving
    ckpt.restore(manager.latest_checkpoint)
    print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)

    try:
        test_mean_loss, test_mean_metrics = compute_loss_and_metrics(test_tf_dataset,
                                                                     keras_model,
                                                                     loss_function,
                                                                     metrics_function)
        print("Test Loss:  {:8.5f}".format(test_mean_loss))

        for metric_name, metric_value in test_mean_metrics.items():
            print("{} {:8.4f}".format(metric_name, metric_value))
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
