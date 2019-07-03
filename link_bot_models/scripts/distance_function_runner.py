#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Input

from link_bot_models import base_model_runner
from link_bot_models.base_model_runner import BaseModelRunner
from link_bot_models.components.distance_function_layer import distance_function_layer
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util


class DistanceFunctionModelRunner(BaseModelRunner):

    def __init__(self, args_dict):
        super(DistanceFunctionModelRunner, self).__init__(args_dict)

        self.n_points = int(self.N / 2)
        self.sigmoid_scale = args_dict['sigmoid_scale']

        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        distance_matrix_layer, layer = distance_function_layer(self.sigmoid_scale, self.n_points, LabelType.Overstretching.name)
        prediction = layer(rope_input)

        self.model_inputs = [rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=prediction)
        losses = {
            LabelType.Overstretching.name: 'binary_crossentropy',
        }
        self.keras_model.compile(optimizer='adam', loss=losses, metrics=['accuracy'])

        self.distance_matrix_model = Model(inputs=self.model_inputs, outputs=distance_matrix_layer.output)

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
        model = DistanceFunctionModelRunner.load(args.checkpoint)
    else:
        args_dict = {
            'sigmoid_scale': 100,
            'N': train_dataset.N,
        }
        args_dict.update(base_model_runner.make_args_dict(args))
        model = DistanceFunctionModelRunner(args_dict)

    model.train(train_dataset, validation_dataset, args.label_types_map, log_path, args)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    model = DistanceFunctionModelRunner.load(args.checkpoint)

    weights = model.keras_model.get_weights()
    conv_kernel = np.squeeze(weights[0])
    conv_bias = np.squeeze(weights[1])
    print(conv_kernel)
    print(conv_bias)
    print(conv_kernel[0, 2] + conv_kernel[2, 0])
    print(conv_kernel[0, 1] + conv_kernel[1, 0] + conv_kernel[1, 2] + conv_kernel[2, 1])

    x = dataset.environments[0].rope_data['rope_configurations'][:1]
    d = model.distance_matrix_model.predict(x)
    print(np.squeeze(d))

    return model.evaluate(dataset, args.label_types_map)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_model_runner.base_parser()

    train_subparser.set_defaults(func=train)
    eval_subparser.set_defaults(func=evaluate)
    # eval_subparser.set_defaults(func=DistanceFunctionModelRunner.evaluate_main)
    show_subparser.set_defaults(func=DistanceFunctionModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
