import json
import pathlib
from typing import Dict

import numpy as np
import progressbar
import tensorflow as tf
from colorama import Fore, Style

from moonshine import experiments_util, loss_on_dicts


class BaseLearnedDynamicsModel(tf.keras.Model):

    def __init__(self, hparams: Dict, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = tf.contrib.checkpoint.NoDependency(hparams)
        self.batch_size = batch_size

    @staticmethod
    def check_validation(val_tf_dataset, loss, net):
        val_losses = []
        for val_x, val_y in progressbar.progressbar(val_tf_dataset):
            val_pred_states = net(val_x)
            # val_y is a dict containing all the state sequences which we should be predicting
            batch_val_loss = loss_on_dicts(loss, dict_true=val_y, dict_pred=val_pred_states)
            val_losses.append(batch_val_loss)
        mean_val_loss = np.mean(val_losses)
        return mean_val_loss

    @classmethod
    def train(cls,
              hparams: Dict,
              train_tf_dataset,
              val_tf_dataset,
              log_path: pathlib.Path,
              seed: int,
              args):
        optimizer = tf.train.AdamOptimizer()
        loss = tf.keras.losses.MeanSquaredError()
        net = cls(hparams=hparams, batch_size=args.batch_size)
        global_step = tf.train.get_or_create_global_step()

        # If we're resuming a checkpoint, there is no new log path
        if args.checkpoint is not None:
            full_log_path = args.checkpoint
        elif args.log:
            full_log_path = pathlib.Path("log_data") / log_path
        else:
            full_log_path = '/tmp'

        ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, net=net)
        manager = tf.train.CheckpointManager(ckpt, full_log_path, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
        elif args.checkpoint:
            print(Fore.RED + "Failed to restore from checkpoint directory {}".format(args.checkpoint) + Fore.RESET)
            print("Did you forget a subdirectory?")
            return

        writer = None
        if args.log is not None:
            print(Fore.CYAN + "Logging to {}".format(full_log_path) + Fore.RESET)

            experiments_util.make_log_dir(full_log_path)

            hparams_path = full_log_path / "hparams.json"
            with open(hparams_path, 'w') as hparams_file:
                hparams['log path'] = str(full_log_path)
                hparams['seed'] = seed
                hparams['batch_size'] = args.batch_size
                hparams['dataset'] = str(args.dataset_dirs)
                hparams_file.write(json.dumps(hparams, indent=2))

            writer = tf.contrib.summary.create_file_writer(logdir=full_log_path)

        def train_loop():
            step = None
            for epoch in range(args.epochs):
                ################
                # train
                ################
                # metrics are averaged across batches in the epoch
                batch_losses = []

                # FIXME: Do I really need this to be x/y style dataset? or maybe I should make the classifier dataset
                #  be an x/y dataset as well
                for train_batch_x, train_batch_y in progressbar.progressbar(train_tf_dataset):
                    step = int(global_step.numpy())

                    with tf.GradientTape() as tape:
                        pred_states = net(train_batch_x)
                        training_batch_loss = loss_on_dicts(loss, dict_true=train_batch_y, dict_pred=pred_states)

                    variables = net.trainable_variables
                    gradients = tape.gradient(training_batch_loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables))
                    batch_losses.append(training_batch_loss.numpy())

                    if args.log:
                        if step % args.log_scalars_every == 0:
                            tf.contrib.summary.scalar("batch loss", training_batch_loss, step=step)

                    ####################
                    # Update global step
                    ####################
                    global_step.assign_add(1)

                training_loss = np.mean(batch_losses)
                print("Epoch: {:5d}, Training loss: {:7.4f}".format(epoch, training_loss))

                ################
                # validation
                ################
                if epoch % args.validation_every == 0:
                    mean_val_loss = cls.check_validation(val_tf_dataset, loss, net)
                    tf.contrib.summary.scalar('validation loss', mean_val_loss, step=int(ckpt.step))
                    print("\t\t\tValidation loss: " + Style.BRIGHT + "{:8.5f}".format(mean_val_loss) + Style.RESET_ALL)

                ################
                # Checkpoint
                ################
                if args.log and epoch % args.save_freq == 0:
                    save_path = manager.save()
                    print(Fore.CYAN + "Step {:6d}: Saved checkpoint {}".format(int(ckpt.step), save_path) + Fore.RESET)

            if args.log:
                save_path = manager.save()
                print(Fore.CYAN + "Step {:6d}: Saved final checkpoint {}".format(step, save_path) + Fore.RESET)

        if args.log:
            with writer.as_default(), tf.contrib.summary.always_record_summaries():
                train_loop()
        else:
            train_loop()

    @classmethod
    def eval(cls, hparams, test_tf_dataset, args):
        net = cls(hparams=hparams, batch_size=args.batch_size)
        ckpt = tf.train.Checkpoint(net=net)
        manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint)
        print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)

        loss = tf.keras.losses.MeanSquaredError()

        test_losses = []
        errors = {}
        for test_x, test_y in test_tf_dataset:
            test_pred_states = net(test_x)
            batch_test_loss = loss_on_dicts(loss, dict_true=test_y, dict_pred=test_pred_states)
            test_losses.append(batch_test_loss)

            for state_key, test_pred_state in test_pred_states.items():
                test_true_state = test_y[state_key]
                test_pred_points = tf.reshape(test_pred_state, [test_pred_state.shape[0], test_pred_state.shape[1], -1, 2])
                test_true_points = tf.reshape(test_true_state, [test_true_state.shape[0], test_true_state.shape[1], -1, 2])
                batch_test_position_error = tf.reduce_mean(tf.linalg.norm(test_pred_points - test_true_points, axis=3), axis=0)
                last_pred_point = test_pred_points[:, -1]
                last_true_point = test_true_points[:, -1]
                final_tail_position_error = tf.reduce_mean(tf.linalg.norm(last_pred_point - last_true_point, axis=2))

                if state_key not in errors:
                    errors[state_key] = {
                        'full_traj': [],
                        'final_step': []
                    }

                errors[state_key]['full_traj'].append(batch_test_position_error)
                errors[state_key]['final_step'].append(final_tail_position_error)

        test_loss = np.mean(test_losses)
        print("Test Loss:  {:8.5f}".format(test_loss))

        for state_key, error_dict in errors.items():
            mean_position_error = np.mean(error_dict['full_traj'])
            mean_final_position_error = np.mean(error_dict['final_step'])
            print("Mean Position Error:" + Style.BRIGHT + "{:8.4f}(m)".format(mean_position_error) + Style.RESET_ALL)
            print("Mean Final Position Error: " + Style.BRIGHT + "{:8.4f}(m)".format(mean_final_position_error) + Style.RESET_ALL)
