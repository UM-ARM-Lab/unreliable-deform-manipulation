#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Lambda, Input

from link_bot_models import base_model
from link_bot_models.base_model import BaseModel
from link_bot_models.components.distance_function_layer import distance_function_layer
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util

distance_function_label_types = [LabelType.Overstretching]


class DistanceFunctionModelRunner(BaseModel):

    def __init__(self, args_dict, N):
        super(DistanceFunctionModelRunner, self).__init__(args_dict, N)

        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        n_points = int(N / 2)
        distance_matrix_layer, layer = distance_function_layer(args_dict['sigmoid_scale'], n_points)
        distance_prediction = layer(rope_input)
        prediction = Lambda(lambda x: x, name='combined_output')(distance_prediction)

        self.model_inputs = [rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=prediction)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        self.distance_matrix_model = Model(inputs=self.model_inputs, outputs=distance_matrix_layer.output)

    def metadata(self, label_types):
        metadata = {
            'sigmoid_scale': self.args_dict['sigmoid_scale'],
        }
        metadata.update(super(DistanceFunctionModelRunner, self).metadata(label_types))
        return metadata

    def violated(self, observations):
        rope_configuration = observations
        inputs_dict = {
            'rope_configuration': rope_configuration,
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)

        self.distance_matrix_model.set_weights(self.keras_model.get_weights())
        predicted_point = self.distance_matrix_model.predict(inputs_dict)

        return predicted_violated, predicted_point


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)

    if args.checkpoint:
        model = DistanceFunctionModelRunner.load(vars(args), args.N)
    else:
        model = DistanceFunctionModelRunner(vars(args), args.N)

    model.train(train_dataset, validation_dataset, distance_function_label_types, args.epochs, log_path)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)

    model = DistanceFunctionModelRunner.load(vars(args), args.N)

    # weights = model.keras_model.get_weights()
    # conv_kernel = np.squeeze(weights[0])
    # conv_bias = np.squeeze(weights[1])
    # print(conv_kernel)
    # print(conv_bias)
    # print(conv_kernel[0, 2] + conv_kernel[2, 0])
    # print(conv_kernel[0, 1] + conv_kernel[1, 0] + conv_kernel[1, 2] + conv_kernel[2, 1])

    # x = dataset.environments[0].rope_data['rope_configurations'][:1]
    # d = model.distance_matrix_model.predict(x)
    # print(np.squeeze(d))

    return model.evaluate(dataset, distance_function_label_types)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_model.base_parser()

    train_subparser.add_argument("--sigmoid-scale", "-s", type=float, default=100)
    train_subparser.set_defaults(func=train)

    eval_subparser.set_defaults(func=evaluate)
    # FIXME: make hyper parameters loaded from metadata
    eval_subparser.add_argument("--sigmoid-scale", "-s", type=float, default=100)

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
