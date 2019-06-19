from __future__ import division, print_function, absolute_import

import json
import os

import keras
import tensorflow as tf
from colorama import Fore
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

from link_bot_pycommon import experiments_util


class BaseModel:

    def __init__(self, args_dict, sdf_shape, N):
        self.args_dict = args_dict
        self.sdf_shape = sdf_shape
        self.N = N

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        set_session(tf.Session(config=config))

        self.keras_model = None

    def metadata(self, label_types):
        raise NotImplementedError()

    def train(self, train_dataset, validation_dataset, label_types, epochs, log_path):
        callbacks = []
        if self.args_dict['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)

            experiments_util.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "metadata.json")
            metadata_file = open(metadata_path, 'w')
            metadata = self.metadata(label_types)
            metadata['log path'] = full_log_path
            metadata_file.write(json.dumps(metadata, indent=2))

            model_filename = os.path.join(full_log_path, "nn.{epoch:02d}.hdf5")

            checkpoint_callback = keras.callbacks.ModelCheckpoint(model_filename, monitor='loss', verbose=0,
                                                                  save_best_only=False, save_weights_only=False, mode='auto',
                                                                  period=1)
            callbacks.append(checkpoint_callback)

        train_generator = train_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
        validation_generator = validation_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
        self.keras_model.fit_generator(train_generator, callbacks=callbacks, validation_data=validation_generator, epochs=epochs)
        self.evaluate(validation_dataset, label_types)

    def evaluate(self, validation_dataset, label_types, display=True):
        generator = validation_dataset.generator_specific_labels(label_types, self.args_dict['batch_size'])
        loss, accuracy = self.keras_model.evaluate_generator(generator)

        if display:
            print("Overall Loss: {:0.3f}".format(float(loss)))
        print("constraint prediction accuracy:\n{}".format(accuracy))

        return loss, accuracy

    @staticmethod
    def load(args_dict):
        keras_model = load_model(args_dict['checkpoint'])
        print(Fore.CYAN + "Restored keras model {}".format(args_dict['checkpoint']) + Fore.RESET)
        return keras_model

    def __str__(self):
        raise NotImplementedError()
