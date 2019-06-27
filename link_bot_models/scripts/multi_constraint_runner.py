#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Concatenate, Lambda, Activation
from keras.models import Model

from link_bot_models import base_model
from link_bot_models.base_model import BaseModelRunner
from link_bot_models.components.distance_function_layer import distance_function_layer
from link_bot_models.components.sdf_function_layer import sdf_function_layer
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util


class MultiConstraintModelRunner(BaseModelRunner):
    def __init__(self, args_dict):
        super(MultiConstraintModelRunner, self).__init__(args_dict)
        self.sdf_shape = args_dict['sdf_shape']
        self.N = args_dict['N']
        self.sigmoid_scale = args_dict['sigmoid_scale']
        self.fc_layer_sizes = args_dict['fc_layer_sizes']
        self.beta = args_dict['beta']

        sdf = Input(shape=[self.sdf_shape[0], self.sdf_shape[1], 1], dtype='float32', name='sdf')
        sdf_gradient = Input(shape=[self.sdf_shape[0], self.sdf_shape[0], 2], dtype='float32', name='sdf_gradient')
        sdf_resolution = Input(shape=[2], dtype='float32', name='sdf_resolution')
        sdf_origin = Input(shape=[2], dtype='float32', name='sdf_origin')  # will be converted to int32 in SDF layer
        sdf_extent = Input(shape=[4], dtype='float32', name='sdf_extent')
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        # Distance Function
        n_points = int(self.N / 2)
        distance_matrix_layer, distance_function = distance_function_layer(self.sigmoid_scale, n_points)
        overstretching_prediction = distance_function(rope_input)

        # SDF Function
        sdf_input_layer, sdf_function = sdf_function_layer(self.sdf_shape, self.fc_layer_sizes, self.beta, self.sigmoid_scale)
        sdf_function_prediction = sdf_function(sdf, sdf_gradient, sdf_resolution, sdf_origin, rope_input)

        # Combine
        concat_predictions = Concatenate(name='all_output')([sdf_function_prediction, overstretching_prediction])
        prediction = Lambda(lambda x: K.sum(x, axis=1, keepdims=True) - K.prod(x, axis=1, keepdims=True), name='combined_output')(concat_predictions)

        self.model_inputs = [sdf, sdf_gradient, sdf_resolution, sdf_origin, sdf_extent, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=[prediction, concat_predictions])
        self.sdf_input_model = Model(inputs=self.model_inputs, outputs=sdf_input_layer.output)

        losses = {
            'combined_output': 'binary_crossentropy',
            # 'all_output': 'binary_crossentropy',
        }
        self.keras_model.compile(optimizer='adam', loss=losses, metrics=['accuracy'])

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

        self.sdf_input_model.set_weights(self.keras_model.get_weights())
        predicted_point = self.sdf_input_model.predict(inputs_dict)

        return predicted_violated, predicted_point


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = MultiConstraintModelRunner.load(args.checkpoint)
    else:
        args_dict = {
            'sdf_shape': sdf_shape,
            'conv_filters': [
                (32, (5, 5)),
                (16, (3, 3)),
            ],
            'beta': 1e-2,
            'fc_layer_sizes': [16, 16],
            'sigmoid_scale': 100,
            'N': train_dataset.N,
        }
        args_dict.update(base_model.make_args_dict(args))
        model = MultiConstraintModelRunner(args_dict)

    model.train(train_dataset, validation_dataset, args.label_types, log_path, args)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_model.base_parser()

    train_subparser.set_defaults(func=train)
    eval_subparser.set_defaults(func=MultiConstraintModelRunner.evaluate_main)
    show_subparser.set_defaults(func=MultiConstraintModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
