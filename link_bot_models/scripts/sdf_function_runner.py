#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model

from link_bot_models.base_model import BaseModel
from link_bot_models.components.sdf_function_layer import sdf_function_layer
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util, link_bot_pycommon

sdf_function_label_types = [LabelType.SDF]


class SDFFunctionModelRunner(BaseModel):
    def __init__(self, args_dict, sdf_shape, N):
        super(SDFFunctionModelRunner, self).__init__(args_dict, N)
        self.sdf_shape = sdf_shape

        sdf = Input(shape=[self.sdf_shape[0], self.sdf_shape[1], 1], dtype='float32', name='sdf')
        sdf_gradient = Input(shape=[self.sdf_shape[0], self.sdf_shape[0], 2], dtype='float32', name='sdf_gradient')
        sdf_resolution = Input(shape=[2], dtype='float32', name='sdf_resolution')
        sdf_origin = Input(shape=[2], dtype='float32', name='sdf_origin')  # will be converted to int32 in SDF layer
        sdf_extent = Input(shape=[4], dtype='float32', name='sdf_extent')
        rope_input = Input(shape=[self.N], dtype='float32', name='rope_configuration')

        self.fc_layer_sizes = [
            16,
            16,
        ]

        self.beta = 1e-2

        sdf_input_layer, sdf_function = sdf_function_layer(sdf_shape, self.fc_layer_sizes, self.beta, args_dict['sigmoid_scale'])
        sdf_function_prediction = sdf_function([sdf, sdf_gradient, sdf_resolution, sdf_origin, rope_input])
        prediction = Lambda(lambda x: x, name='combined_output')(sdf_function_prediction)

        self.model_inputs = [sdf, sdf_gradient, sdf_resolution, sdf_origin, sdf_extent, rope_input]
        self.keras_model = Model(inputs=self.model_inputs, outputs=prediction)
        self.sdf_input_model = Model(inputs=self.model_inputs, outputs=sdf_input_layer.output)

        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def metadata(self, label_types):
        extra_metadata = {
            'beta': self.beta,
            'sdf_shape': self.sdf_shape,
            'sigmoid_scale': self.args_dict['sigmoid_scale'],
            'hidden_layer_dims': self.fc_layer_sizes,
        }
        extra_metadata.update(super(SDFFunctionModelRunner, self).metadata(label_types))
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


class EvaluateResult:

    def __init__(self, rope_configuration, predicted_point, predicted_violated, true_violated):
        self.rope_configuration = rope_configuration
        self.predicted_point = predicted_point
        self.predicted_violated = predicted_violated
        self.true_violated = true_violated


def test_single_prediction(sdf_data, model, threshold, rope_configuration):
    rope_configuration = rope_configuration.reshape(-1, 6)
    predicted_violated, predicted_point = model.violated(rope_configuration, sdf_data)
    predicted_point = predicted_point.squeeze()
    rope_configuration = rope_configuration.squeeze()
    head_x = rope_configuration[4]
    head_y = rope_configuration[5]
    row_col = link_bot_pycommon.point_to_sdf_idx(head_x, head_y, sdf_data.resolution, sdf_data.origin)
    true_violated = sdf_data.sdf[row_col] < threshold

    result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, true_violated)
    return result


def test_predictions(model, environment):
    rope_configurations = environment.rope_data['rope_configurations']
    constraint_labels = environment.rope_data['constraints']

    predicted_violateds, predicted_points = model.violated(rope_configurations, environment.sdf_data)

    m = rope_configurations.shape[0]
    results = np.ndarray([m], dtype=EvaluateResult)
    for i in range(m):
        rope_configuration = rope_configurations[i]
        predicted_point = predicted_points[i]
        predicted_violated = predicted_violateds[i]
        constraint_label = constraint_labels[i]
        result = EvaluateResult(rope_configuration, predicted_point, predicted_violated, constraint_label)
        results[i] = result
    return results


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = SDFFunctionModelRunner.load(vars(args), sdf_shape, args.N)
    else:
        model = SDFFunctionModelRunner(vars(args), sdf_shape, args.N)

    model.train(train_dataset, validation_dataset, sdf_function_label_types, args.epochs, log_path)
    model.evaluate(validation_dataset, sdf_function_label_types)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    sdf_shape = dataset.sdf_shape

    model = SDFFunctionModelRunner.load(vars(args), sdf_shape, args.N)

    return model.evaluate(dataset, sdf_function_label_types)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("train_dataset", help="dataset (json file)")
    train_subparser.add_argument("validation_dataset", help="dataset (json file)")
    train_subparser.add_argument("--batch-size", "-b", type=int, default=100)
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=50)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--random-init", action='store_true')
    train_subparser.add_argument("--plot", action='store_true')
    train_subparser.add_argument("--sigmoid-scale", "-s", type=float, default=100)
    train_subparser.add_argument("--skip-validation", action='store_true')
    train_subparser.add_argument("--early-stopping", action='store_true')
    train_subparser.add_argument("--val-acc-threshold", type=float, default=None)
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (json file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.add_argument("--batch-size", "-b", type=int, default=100)
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
