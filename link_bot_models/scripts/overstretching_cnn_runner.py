#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense

import link_bot_pycommon.link_bot_sdf_tools
from link_bot_models import base_model_runner
from link_bot_models.base_model_runner import BaseModelRunner
from link_bot_models.components.simple_cnn_layer import simple_cnn_layer
from link_bot_data.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util, link_bot_pycommon


class OverstretchingCNNModelRunner(BaseModelRunner):

    def __init__(self, args_dict):
        super(OverstretchingCNNModelRunner, self).__init__(args_dict)
        self.sdf_shape = args_dict['sdf_shape']

        rope_image = Input(shape=(self.sdf_shape[0], self.sdf_shape[1], 3), dtype='float32', name='rope_image')

        self.conv_filters = args_dict['conv_filters']
        self.fc_layer_sizes = args_dict['fc_layer_sizes']

        cnn_output = simple_cnn_layer(self.conv_filters, self.fc_layer_sizes)(rope_image)
        predictions = Dense(1, activation='sigmoid', name='combined_output')(cnn_output)

        self.model_inputs = [rope_image]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def violated(self, observations, sdf_data):
        rope_configurations = observations
        rope_images = link_bot_pycommon.link_bot_sdf_tools.make_rope_images(sdf_data, rope_configurations)
        inputs_dict = {
            'rope_image': rope_images,
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)
        return predicted_violated


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = OverstretchingCNNModelRunner.load(args.checkpoint)
    else:
        args_dict = {
            'sdf_shape': sdf_shape,
            'conv_filters': [
                (32, (7, 7)),
                (16, (5, 5)),
            ],
            'fc_layer_sizes': [
                32, 32
            ],
            'N': train_dataset.N,
        }
        args_dict.update(base_model_runner.make_args_dict(args))
        model = OverstretchingCNNModelRunner(args_dict)

    model.train(train_dataset, validation_dataset, args.label_types, log_path, args)


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_model_runner.base_parser()
    train_subparser.set_defaults(func=train)
    eval_subparser.set_defaults(func=OverstretchingCNNModelRunner.evaluate_main)
    show_subparser.set_defaults(func=OverstretchingCNNModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
