from __future__ import division, print_function, absolute_import

import json
import os

import keras
import tensorflow as tf
from colorama import Fore
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model
from keras.models import load_model

from link_bot_pycommon import experiments_util


class ConstraintCNN:

    def __init__(self, args_dict, sdf_shape, N):
        self.args_dict = args_dict
        self.N = N

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=config))

        sdf_input = Input(shape=(sdf_shape[0], sdf_shape[1], 1), dtype='float32', name='sdf')
        rope_input = Input(shape=(N,), dtype='float32', name='rope_configuration')

        self.conv_filters = [
            (64, (3, 3)),
            (64, (3, 3)),
        ]

        self.fc_layer_sizes = [
            128,
            128,
        ]

        conv_h = sdf_input
        for conv_filter in self.conv_filters:
            n_filters = conv_filter[0]
            filter_size = conv_filter[1]
            conv_z = Conv2D(n_filters, filter_size)(conv_h)
            conv_h = MaxPool2D(2)(conv_z)
        conv_output = Flatten()(conv_h)

        concat = keras.layers.concatenate([conv_output, rope_input])
        fc_h = concat
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        predictions = Dense(1, activation='sigmoid', name='combined_output')(fc_h)

        # This creates a model that includes
        self.keras_model = Model(inputs=[sdf_input, rope_input], outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def metadata(self, label_types):
        metadata = {
            'tf_version': str(tf.__version__),
            'keras_version': str(keras.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'conv_filters': self.conv_filters,
            'label_type': [label_type.name for label_type in label_types],
            'fc_layer_sizes': self.fc_layer_sizes,
            'commandline': self.args_dict['commandline'],
        }
        return metadata

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

    def violated(self, observations, sdf_data):
        x = [observations, sdf_data]
        predicted_violated = self.keras_model.predict(x)
        return predicted_violated

    @staticmethod
    def load(args_dict):
        keras_model = load_model(args_dict['checkpoint'])
        print(Fore.CYAN + "Restored keras model {}".format(args_dict['checkpoint']) + Fore.RESET)
        return keras_model

    def __str__(self):
        return "keras constraint cnn"
