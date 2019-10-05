from __future__ import division, print_function, absolute_import

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_classifiers import my_viz
from link_bot_classifiers.callbacks import StopAtAccuracy, DebugCallback
from link_bot_pycommon import experiments_util
from video_prediction.datasets import dataset_utils


def base_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train")
    eval_parser = subparsers.add_parser("eval")
    show_parser = subparsers.add_parser("show")

    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)

    train_parser.add_argument("input_dir", help="directory of tfrecords")
    train_parser.add_argument("dataset_hparams_dict", type=str, help="json file of hyperparameters")
    train_parser.add_argument("--dataset-hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    train_parser.add_argument("--batch-size", "-b", type=int, default=32)
    train_parser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_parser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=50)
    train_parser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_parser.add_argument("--debug", action='store_true')
    train_parser.add_argument("--balance", action='store_true')
    train_parser.add_argument("--validation", action='store_true')
    train_parser.add_argument("--early-stopping", action='store_true')
    train_parser.add_argument("--val-acc-threshold", type=float, default=None)

    eval_parser.add_argument("input_dir", help="directory of tfrecords")
    eval_parser.add_argument("dataset_hparams_dict", type=str, help="json file of hyperparameters")
    eval_parser.add_argument("checkpoint")
    eval_parser.add_argument("--dataset-hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    eval_parser.add_argument("--batch-size", "-b", type=int, default=32)

    show_parser.add_argument("checkpoint", help="eval the *.ckpt name")
    show_parser.add_argument("input_dir", help="directory of tfrecords")
    show_parser.add_argument("dataset_hparams_dict", type=str, help="json file of hyperparameters")
    show_parser.add_argument("--dataset-hparams", type=str, help="a string of comma separated list of dataset hyperparameters")

    def run_parser():
        commandline = ' '.join(sys.argv)

        args = parser.parse_args()
        args.commandline = commandline

        np.random.seed(args.seed)
        tf.random.set_random_seed(args.seed)

        if args == argparse.Namespace():
            parser.print_usage()
        else:
            args.func(args)

    parser.run = run_parser

    return parser, train_parser, eval_parser, show_parser


def make_args_dict(args):
    return {
        'seed': args.seed,
        'batch_size': args.batch_size,
    }


class BaseClassifierRunner:

    def __init__(self, args_dict):
        """
        :param args_dict: Everything needed to define the architecture
        """
        self.args_dict = args_dict
        self.seed = args_dict['seed']
        self.batch_size = args_dict['batch_size']
        self.N = args_dict['N']

        self.initial_epoch = 0

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        set_session(tf.Session(config=config))

        self.keras_model = None

    def base_metadata(self, args):
        metadata = {
            'tf_version': str(tf.__version__),
            'checkpoint': args.checkpoint,
            'commandline': args.commandline,
            'args_dict': self.args_dict,
        }
        return metadata

    def train(self, train_dataset, train_tf_dataset, val_dataset, val_tf_dataset, log_path, args):
        callbacks = []
        if args.log is not None:
            full_log_path = os.path.join("log_data", log_path)

            print(Fore.CYAN + "Logging to {}".format(full_log_path) + Fore.RESET)

            experiments_util.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "metadata.json")
            with open(metadata_path, 'w') as metadata_file:
                metadata = self.base_metadata(args)
                metadata['log path'] = full_log_path
                metadata_file.write(json.dumps(metadata, indent=2))

            model_filename = os.path.join(full_log_path, "nn.{epoch:02d}.hdf5")

            checkpoint_callback = ModelCheckpoint(model_filename, monitor='loss', save_weights_only=True)
            callbacks.append(checkpoint_callback)

            tensorboard = TensorBoard(log_dir=full_log_path)
            callbacks.append(tensorboard)

            val_acc_threshold = args.val_acc_threshold
            if val_acc_threshold is not None:
                if args.validation:
                    raise ValueError("Validation dataset must be provided in order to use this monitor")
                if val_acc_threshold < 0 or val_acc_threshold > 1:
                    raise ValueError("val_acc_threshold {} must be between 0 and 1 inclusive".format(val_acc_threshold))
                stop_at_accuracy = StopAtAccuracy(val_acc_threshold)
                callbacks.append(stop_at_accuracy)

            if args.early_stopping:
                if args.validation:
                    raise ValueError("Validation dataset must be provided in order to use this monitor")
                early_stopping = EarlyStopping(monitor='val_acc', patience=5, min_delta=0.001, verbose=args.verbose)
                callbacks.append(early_stopping)

            if args.debug:
                callbacks.append(DebugCallback())

        steps_per_epoch = train_dataset.num_examples_per_epoch() // args.batch_size
        val_steps_per_epoch = val_dataset.num_examples_per_epoch() // args.batch_size

        if not args.validation:
            val_tf_dataset = None
            val_steps_per_epoch = None

        self.keras_model.fit(x=train_tf_dataset,
                             y=None,
                             callbacks=callbacks,
                             initial_epoch=self.initial_epoch,
                             steps_per_epoch=steps_per_epoch,
                             validation_data=val_tf_dataset,
                             validation_steps=val_steps_per_epoch,
                             epochs=args.epochs,
                             verbose=True)

    def evaluate(self, args, dataset, tf_dataset, display=True):
        steps_per_epoch = dataset.num_examples_per_epoch() // args.batch_size
        metrics = self.keras_model.evaluate(tf_dataset, steps=steps_per_epoch)
        if display:
            print("Validation:")
            for name, metric in zip(self.keras_model.metrics_names, metrics):
                print("{}: {:4.4f}".format(name, metric))

        return self.keras_model.metrics_names, metrics

    @classmethod
    def load(cls, checkpoint):
        checkpoint_path = pathlib.Path(checkpoint)
        metadata_path = checkpoint_path.parent / 'metadata.json'
        metadata = json.load(open(metadata_path, 'r'))
        args_dict = metadata['args_dict']
        model = cls(args_dict)

        basename = os.path.basename(os.path.splitext(checkpoint)[0])
        initial_epoch = int(basename[3:])
        model.initial_epoch = initial_epoch
        model.keras_model.load_weights(checkpoint)
        print(Fore.CYAN + "Restored keras model {}".format(checkpoint) + Fore.RESET)
        return model

    @classmethod
    def evaluate_main(cls, args):
        model = cls.load(args.checkpoint)

        dataset, tf_dataset = dataset_utils.get_dataset(args.input_dir,
                                                        'link_bot',
                                                        args.dataset_hparams_dict,
                                                        args.dataset_hparams,
                                                        mode='val',
                                                        epochs=1,
                                                        seed=args.seed,
                                                        batch_size=1)

        return model.evaluate(args, dataset, tf_dataset)

    @classmethod
    def show(cls, args):
        model = cls.load(args.checkpoint)
        print(model.keras_model.summary())
        path = pathlib.Path(args.checkpoint)
        names = [pathlib.Path(part).stem for part in path.parts if part != '/' and part != 'log_data']
        image_filename = 'img~' + "~".join(names) + '.png'

        names = [weight.name for layer in model.keras_model.layers for weight in layer.weights]
        weights = model.keras_model.get_weights()

        for name, weight in zip(names, weights):
            print(name, weight)

        my_viz.plot_model(model.keras_model, to_file=image_filename, show_shapes=True)
