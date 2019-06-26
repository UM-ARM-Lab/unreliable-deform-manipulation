from __future__ import division, print_function, absolute_import

import argparse
import json
import os
import pathlib

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from colorama import Fore
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import load_model

from link_bot_models.callbacks import StopAtAccuracy
from link_bot_models.components.bias_layer import BiasLayer
from link_bot_models.components.distance_matrix_layer import DistanceMatrix
from link_bot_models.components.out_of_bounds_regularization import OutOfBoundsRegularizer
from link_bot_models.components.sdf_lookup import SDFLookup
from link_bot_pycommon import experiments_util

custom_objects = {
    'BiasLayer': BiasLayer,
    'SDFLookup': SDFLookup,
    'DistanceMatrix': DistanceMatrix,
    'OutOfBoundsRegularizer': OutOfBoundsRegularizer,
}


def load_keras_model_only(args):
    keras_model = load_model(args.checkpoint, custom_objects=custom_objects)
    print(Fore.CYAN + "Restored keras model {}".format(args.checkpoint) + Fore.RESET)
    return keras_model


def show(args):
    keras_model = load_keras_model_only(args)
    print(keras_model.summary())
    path = pathlib.Path(args.checkpoint)
    names = [pathlib.Path(part).stem for part in path.parts if part != '/' and part != 'log_data']
    image_filename = "~".join(names) + '.png'
    keras.utils.plot_model(keras_model, to_file=image_filename)


def base_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    eval_subparser = subparsers.add_parser("eval")
    show_subparser = subparsers.add_parser("show")

    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)

    train_subparser.add_argument("train_dataset", help="dataset (json file)")
    train_subparser.add_argument("validation_dataset", help="dataset (json file)")
    train_subparser.add_argument("--batch-size", "-b", type=int, default=100)
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=50)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--plot", action='store_true')
    train_subparser.add_argument("--validation-steps", type=int, default=-1)
    train_subparser.add_argument("--early-stopping", action='store_true')
    train_subparser.add_argument("--val-acc-threshold", type=float, default=None)

    eval_subparser.add_argument("dataset", help="dataset (json file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.add_argument("--batch-size", "-b", type=int, default=100)

    show_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    show_subparser.set_defaults(func=show)

    return parser, train_subparser, eval_subparser, show_subparser


class BaseModel:

    def __init__(self, args_dict, N):
        self.args_dict = args_dict
        self.N = N
        self.initial_epoch = 0

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        set_session(tf.Session(config=config))

        self.keras_model = None

    def metadata(self, label_types):
        metadata = {
            'tf_version': str(tf.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'label_type': [label_type.name for label_type in label_types],
            'commandline': self.args_dict['commandline'],
        }

        return metadata

    def train(self, train_dataset, validation_dataset, label_types, epochs, log_path):
        callbacks = []
        if self.args_dict['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)

            print(Fore.CYAN + "Logging to {}".format(full_log_path) + Fore.RESET)

            experiments_util.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "metadata.json")
            metadata_file = open(metadata_path, 'w')
            metadata = self.metadata(label_types)
            metadata['log path'] = full_log_path
            metadata_file.write(json.dumps(metadata, indent=2))

            model_filename = os.path.join(full_log_path, "nn.{epoch:02d}.hdf5")

            checkpoint_callback = ModelCheckpoint(model_filename, monitor='loss')
            callbacks.append(checkpoint_callback)

            tensorboard = TensorBoard(log_dir=full_log_path)
            callbacks.append(tensorboard)

            val_acc_threshold = self.args_dict['val_acc_threshold']
            if val_acc_threshold is not None:
                if self.args_dict['validation_steps']:
                    raise ValueError("Validation dataset must be provided in order to use this monitor")
                if val_acc_threshold < 0 or val_acc_threshold > 1:
                    raise ValueError("val_acc_threshold {} must be between 0 and 1 inclusive".format(val_acc_threshold))
                stop_at_accuracy = StopAtAccuracy(val_acc_threshold)
                callbacks.append(stop_at_accuracy)

            if self.args_dict['early_stopping']:
                if self.args_dict['validation_steps']:
                    raise ValueError("Validation dataset must be provided in order to use this monitor")
                early_stopping = EarlyStopping(monitor='val_acc', patience=5, min_delta=0.001, verbose=True)
                callbacks.append(early_stopping)

            # callbacks.append(DebugCallback())

        train_generator = train_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])

        if self.args_dict['validation_steps'] == -1:
            validation_generator = validation_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
            validation_steps = None
        elif self.args_dict['validation_steps'] > 0:
            validation_generator = validation_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
            validation_steps = self.args_dict['validation_steps']
        else:
            validation_generator = None
            validation_steps = None

        history = self.keras_model.fit_generator(train_generator,
                                                 callbacks=callbacks,
                                                 validation_data=validation_generator,
                                                 initial_epoch=self.initial_epoch,
                                                 validation_steps=validation_steps,
                                                 epochs=epochs)

        if self.args_dict['plot']:
            plt.figure()
            plt.title("Loss")
            plt.plot(history.history['loss'])

            plt.figure()
            plt.title("Accuracy")
            plt.plot(history.history['acc'])

        if self.args_dict['validation_steps']:
            self.evaluate(validation_dataset, label_types)

    def evaluate(self, validation_dataset, label_types, display=True):
        generator = validation_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
        loss, accuracy = self.keras_model.evaluate_generator(generator)

        if display:
            print("Validation:")
            print("Overall Loss: {:0.3f}".format(float(loss)))
            print("constraint prediction accuracy:\n{:5.2f}".format(accuracy * 100))

        return loss, accuracy

    @classmethod
    def load(cls, args_dict, *args):
        model = cls(args_dict, *args)

        basename = os.path.basename(os.path.splitext(args_dict['checkpoint'])[0])
        initial_epoch = int(basename[3:])
        keras_model = load_model(args_dict['checkpoint'], custom_objects=custom_objects)
        model.keras_model = keras_model
        model.initial_epoch = initial_epoch
        print(Fore.CYAN + "Restored keras model {}".format(args_dict['checkpoint']) + Fore.RESET)
        return model
