from __future__ import division, print_function, absolute_import

import json
import os

import keras
import numpy as np
import tensorflow as tf
from colorama import Fore
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
from keras.models import load_model

from link_bot_models.multi_environment_datasets import LabelType
from link_bot_pycommon import experiments_util


class ConstraintCNN:

    def __init__(self, args_dict, sdf_shape, N):
        self.args_dict = args_dict
        self.N = N

        self.label_type = self.args_dict['label_type']
        if self.label_type == LabelType.SDF:
            self.label_mask = np.array([1, 0], dtype=np.int)
        elif self.label_type == LabelType.Overstretching:
            self.label_mask = np.array([0, 1], dtype=np.int)
        elif self.label_type == LabelType.SDF_and_Overstretching:
            self.label_mask = np.array([1, 1], dtype=np.int)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=config))

        sdf_input = Input(shape=(sdf_shape[0], sdf_shape[0], 1), dtype='float32', name='sdf_input')
        rope_input = Input(shape=(N,), dtype='float32', name='rope_input')

        self.conv_filters = [
            (16, (3, 3)),
            (16, (3, 3)),
        ]

        self.fc_layer_sizes = [
            16,
            8,
        ]

        conv_h = sdf_input
        for conv_filter in self.conv_filters:
            n_filters = conv_filter[0]
            filter_size = conv_filter[1]
            conv_h = Conv2D(n_filters, filter_size)(conv_h)
        conv_output = Flatten()(conv_h)

        concat = keras.layers.concatenate([conv_output, rope_input])
        fc_h = concat
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        predictions = Dense(1, activation='sigmoid')(fc_h)

        # This creates a model that includes
        # the Input layer and three Dense layers
        self.keras_model = Model(inputs=[sdf_input, rope_input], outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def metadata(self):
        metadata = {
            'tf_version': str(tf.__version__),
            'keras_version': str(keras.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'conv_filters': self.conv_filters,
            'label_type': str(self.label_type),
            'fc_layer_sizes': self.fc_layer_sizes,
            'commandline': self.args_dict['commandline'],
        }
        return metadata

    def prepare_data(self, x, y):
        rope_inputs = x[0]
        sdf_inputs = np.expand_dims(x[1], axis=3)
        labels = np.any(y[0] * self.label_mask, axis=1).astype(np.float32)
        return {'sdf_input': sdf_inputs, 'rope_input': rope_inputs}, labels

    def train(self, train_x, train_y, validation_x, validation_y, epochs, log_path, **kwargs):
        """
        Wrapper around model.fit

        :param train_x: first dimension is each type of input and the second dimension is examples, following dims are the data
        :param train_y: first dimension is type of label, second dimension is examples, following dims are the labels
        :param validation_x: ''
        :param validation_y: ''
        :param epochs: number of times to run through the full training set
        :param log_path:
        :param kwargs:
        :return: whether the training process was interrupted early (by Ctrl+C)
        """
        train_inputs, train_labels = self.prepare_data(train_x, train_y)
        validation_inputs, validation_labels = self.prepare_data(validation_x, validation_y)

        callbacks = []
        if self.args_dict['log'] is not None:
            full_log_path = os.path.join("log_data", log_path)

            experiments_util.make_log_dir(full_log_path)

            metadata_path = os.path.join(full_log_path, "metadata.json")
            metadata_file = open(metadata_path, 'w')
            metadata = self.metadata()
            metadata['log path'] = full_log_path
            metadata_file.write(json.dumps(metadata, indent=2))

            model_filename = os.path.join(full_log_path, "nn.{epoch:02d}.hdf5")

            checkpoint_callback = keras.callbacks.ModelCheckpoint(model_filename, monitor='val_loss', verbose=0,
                                                                  save_best_only=False, save_weights_only=False, mode='auto',
                                                                  period=1)
            callbacks.append(checkpoint_callback)

        self.keras_model.fit(x=train_inputs,
                             y=train_labels,
                             callbacks=callbacks,
                             validation_data=(validation_inputs, validation_labels),
                             batch_size=self.args_dict['batch_size'],
                             epochs=epochs)
        self.evaluate(validation_x, validation_y)

    def evaluate(self, eval_x, eval_y, display=True):
        inputs, labels = self.prepare_data(eval_x, eval_y)
        loss, accuracy = self.keras_model.evaluate(inputs, labels)

        if display:
            print("Overall Loss: {:0.3f}".format(float(loss)))
            print("constraint prediction accuracy:\n{}".format(accuracy))

        return loss, accuracy

    def violated(self, observations, sdf_data):
        x = [observations, sdf_data]
        predicted_violated = self.keras_model.predict(x)
        return predicted_violated

    def load(self):
        self.keras_model = load_model(self.args_dict['checkpoint'])
        print(Fore.CYAN + "Restored keras model {}".format(self.args_dict['checkpoint']) + Fore.RESET)

    def __str__(self):
        return "keras constraint cnn"

    def build_feed_dict(self, x, y, **kwargs):
        raise NotImplementedError("keras based models don't use this method")
