from __future__ import division, print_function, absolute_import

import json
import os

import keras
import numpy as np
import tensorflow as tf
from colorama import Fore
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model
from keras.models import load_model

from link_bot_models.label_types import LabelType
from link_bot_pycommon import experiments_util
from link_bot_pycommon import link_bot_pycommon


class ConstraintRasterCNN:

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
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        set_session(tf.Session(config=config))

        input = Input(shape=(sdf_shape[0], sdf_shape[1], 4), dtype='float32', name='input')

        self.conv_filters = [
            (16, (3, 3)),
            (16, (3, 3)),
        ]

        self.fc_layer_sizes = [
            16,
            16,
        ]

        conv_h = input
        for conv_filter in self.conv_filters:
            n_filters = conv_filter[0]
            filter_size = conv_filter[1]
            conv_z = Conv2D(n_filters, filter_size)(conv_h)
            conv_h = MaxPool2D(2)(conv_z)

        conv_output = Flatten()(conv_h)

        fc_h = conv_output
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        predictions = Dense(1, activation='combined_output')(fc_h)

        self.keras_model = Model(inputs=input, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def metadata(self):
        metadata = {
            'tf_version': str(tf.__version__),
            'keras_version': str(keras.__version__),
            'seed': self.args_dict['seed'],
            'checkpoint': self.args_dict['checkpoint'],
            'N': self.N,
            'conv_filters': self.conv_filters,
            'fc_layer_sizes': self.fc_layer_sizes,
            'label_type': str(self.label_type),
            'commandline': self.args_dict['commandline'],
        }
        return metadata

    def prepare_data(self, x, y):
        rope_inputs = x[0]
        sdf = np.expand_dims(x[1], axis=3)
        sdf_origin = x[3]
        sdf_resolution = x[4]
        n_rope_points = int(rope_inputs.shape[1] // 2)
        rope_imgs = np.zeros([sdf.shape[0], sdf.shape[1], sdf.shape[2], n_rope_points])
        for i, (origin, resolution) in enumerate(zip(sdf_origin, sdf_resolution)):
            for j in range(n_rope_points):
                px = rope_inputs[i, 2 * j]
                py = rope_inputs[i, 2 * j + 1]
                row, col = link_bot_pycommon.point_to_sdf_idx(px, py, resolution, origin)
                rope_imgs[i, row, col, j] = 1

        # put in range from 0 - 1 so all channels have the same range
        sdf = sdf + np.min(sdf)
        sdf = sdf / np.max(sdf)
        inputs = np.concatenate((sdf, rope_imgs), axis=3)
        labels = np.any(y[0] * self.label_mask, axis=1).astype(np.float32)
        return inputs, labels

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
                                                                  period=5)
            callbacks.append(checkpoint_callback)

        datagen = ImageDataGenerator()
        self.keras_model.fit_generator(datagen.flow(x=train_inputs, y=train_labels, batchsize=self.args['batch_size']),
                                       callbacks=callbacks,
                                       validation_data=(validation_inputs, validation_labels),
                                       workers=4,
                                       epochs=epochs)
        # self.keras_model.fit(x=train_inputs,
        #                      y=train_labels,
        #                      callbacks=callbacks,
        #                      validation_data=(validation_inputs, validation_labels),
        #                      batch_size=self.args_dict['batch_size'],
        #                      epochs=epochs)
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
