#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras.layers import Input, Concatenate, Dense
from keras.models import Model

from link_bot_models import base_model
from link_bot_models.base_model import BaseModelRunner
from link_bot_models.components.simple_cnn_layer import simple_cnn_layer
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util


class SimpleCNNModelRunner(BaseModelRunner):

    def __init__(self, args_dict):
        super(SimpleCNNModelRunner, self).__init__(args_dict)
        self.sdf_shape = args_dict['sdf_shape']

        sdf_input = Input(shape=(self.sdf_shape[0], self.sdf_shape[1], 1), dtype='float32', name='sdf')
        rope_input = Input(shape=(self.N,), dtype='float32', name='rope_configuration')

        self.conv_filters = args_dict['conv_filters']
        self.fc_layer_sizes = args_dict['fc_layer_sizes']

        cnn_output = simple_cnn_layer(self.conv_filters, [])(sdf_input)
        concat = Concatenate()([cnn_output, rope_input])

        fc_h = concat
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        prediction = Dense(1, activation='sigmoid', name='combined_output')(fc_h)

        # This creates a model that includes
        self.model_inputs = [sdf_input, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=prediction)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def violated(self, observations, sdf_data):
        m = observations.shape[0]
        rope_configuration = observations
        sdf = np.tile(np.expand_dims(sdf_data.sdf, axis=2), [m, 1, 1, 1])
        sdf_gradient = np.tile(sdf_data.gradient, [m, 1, 1, 1])
        sdf_origin = np.tile(sdf_data.origin, [m, 1])
        sdf_resolution = np.tile(sdf_data.resolution, [m, 1])
        sdf_extent = np.tile(sdf_data.extent, [m, 1])
        inputs_dict = {
            'rope_configuration': rope_configuration,
            'sdf': sdf,
            'sdf_gradient': sdf_gradient,
            'sdf_origin': sdf_origin,
            'sdf_resolution': sdf_resolution,
            'sdf_extent': sdf_extent
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)
        return predicted_violated


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = SimpleCNNModelRunner.load(args.checkpoint)
    else:
        args_dict = {
            'sdf_shape': sdf_shape,
            'conv_filters': [
                (64, (7, 7)),
                (32, (5, 5)),
                (16, (3, 3)),
                (16, (3, 3)),
            ],
            'fc_layer_sizes': [
                32,
            ],
            'N': train_dataset.N,
        }
        args_dict.update(base_model.make_args_dict(args))
        model = SimpleCNNModelRunner(args_dict)

    model.train(train_dataset, validation_dataset, args.label_types, log_path, args)


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_model.base_parser()

    train_subparser.set_defaults(func=train)
    eval_subparser.set_defaults(func=SimpleCNNModelRunner.evaluate_main)
    show_subparser.set_defaults(func=SimpleCNNModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
