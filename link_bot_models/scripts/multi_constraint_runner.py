#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Concatenate, Dense, Dot
from keras.models import Model

from link_bot_models import base_model
from link_bot_models.base_model import BaseModelRunner
from link_bot_models.components.distance_function_layer import distance_function_layer
from link_bot_models.components.sdf_function_layer import sdf_function_layer
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util

multi_constraint_label_types = [LabelType.SDF, LabelType.Overstretching]


class MultiConstraintModelRunner(BaseModelRunner):
    def __init__(self, args_dict, sdf_shape, N):
        super(MultiConstraintModelRunner, self).__init__(args_dict, N)
        self.sdf_shape = sdf_shape

        sdf = Input(shape=[self.sdf_shape[0], self.sdf_shape[1], 1], dtype='float32', name='sdf')
        sdf_gradient = Input(shape=[self.sdf_shape[0], self.sdf_shape[0], 2], dtype='float32', name='sdf_gradient')
        sdf_resolution = Input(shape=[2], dtype='float32', name='sdf_resolution')
        sdf_origin = Input(shape=[2], dtype='float32', name='sdf_origin')  # will be converted to int32 in SDF layer
        sdf_extent = Input(shape=[4], dtype='float32', name='sdf_extent')
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        #####################
        # Distance Function #
        #####################
        n_points = int(N / 2)
        distance_matrix_layer, distance_function = distance_function_layer(args_dict['sigmoid_scale'], n_points)
        overstretching_prediction = distance_function(rope_input)

        ################
        # SDF Function #
        ################
        self.fc_layer_sizes = [16, 16]

        self.beta = 1e-2
        sdf_input_layer, sdf_function = sdf_function_layer(sdf_shape, self.fc_layer_sizes, self.beta, args_dict['sigmoid_scale'])
        sdf_function_prediction = sdf_function(sdf, sdf_gradient, sdf_resolution, sdf_origin, rope_input)

        ###########
        # Combine #
        ###########
        self.decision_fc_layer_sizes = [16, 16]

        decision_fc_h = rope_input
        for fc_layer_size in self.decision_fc_layer_sizes:
            decision_fc_h = Dense(fc_layer_size, activation='relu')(decision_fc_h)
        prediction_weights = Dense(2, activation='softmax', name='prediction_weights')(decision_fc_h)

        concat_predictions = Concatenate(name='all_output')([sdf_function_prediction, overstretching_prediction])
        prediction = Dot(axes=1, name='combined_output')([prediction_weights, concat_predictions])

        self.model_inputs = [sdf, sdf_gradient, sdf_resolution, sdf_origin, sdf_extent, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=[prediction, concat_predictions])
        self.sdf_input_model = Model(inputs=self.model_inputs, outputs=sdf_input_layer.output)
        self.decision_model = Model(inputs=self.model_inputs, outputs=prediction_weights)

        losses = {
            'combined_output': 'binary_crossentropy',
            'all_output': 'categorical_crossentropy',
        }
        self.keras_model.compile(optimizer='adam', loss=losses, metrics=['accuracy'])

    def metadata(self, label_types):
        extra_metadata = {
            'beta': self.beta,
            'sdf_shape': self.sdf_shape,
            'sigmoid_scale': self.args_dict['sigmoid_scale'],
            'hidden_layer_dims': self.fc_layer_sizes,
            'decision_hidden_layer_dims': self.decision_fc_layer_sizes,
        }
        extra_metadata.update(super(MultiConstraintModelRunner, self).metadata(label_types))
        return extra_metadata

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
        model = MultiConstraintModelRunner.load(vars(args), sdf_shape, args.N)
    else:
        model = MultiConstraintModelRunner(vars(args), sdf_shape, args.N)

    model.train(train_dataset, validation_dataset, multi_constraint_label_types, args.epochs, log_path)
    model.evaluate(validation_dataset, multi_constraint_label_types)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    sdf_shape = dataset.sdf_shape

    model = MultiConstraintModelRunner.load(vars(args), sdf_shape, args.N)

    return model.evaluate(dataset, multi_constraint_label_types)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Add arguments that all models need
    parser, train_subparser, eval_subparser, show_subparser = base_model.base_parser()

    # Custom arguments for training
    train_subparser.add_argument("--sigmoid-scale", "-s", type=float, default=100)
    train_subparser.set_defaults(func=train)

    # Custom arguments for evaluation
    eval_subparser.set_defaults(func=evaluate)

    args = parser.parse_args()
    commandline = ' '.join(sys.argv)
    args.commandline = commandline

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
