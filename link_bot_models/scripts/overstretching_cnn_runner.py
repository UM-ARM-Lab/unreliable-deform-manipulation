#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Input, Lambda

from link_bot_models.base_model import BaseModel
from link_bot_models.components.simple_cnn_model import SimpleCNNModel
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util, link_bot_pycommon

overstretching_label_types = [LabelType.Overstretching]


class OverstretchingCNNModelRunner(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(OverstretchingCNNModelRunner, self).__init__(args_dict, N)
        self.sdf_shape = sdf_shape

        rope_image = Input(shape=(sdf_shape[0], sdf_shape[1], 3), dtype='float32', name='rope_image')

        self.conv_filters = [
            (32, (7, 7)),
            (16, (5, 5)),
        ]

        self.fc_layer_sizes = [
            32,
            32,
        ]

        layer = SimpleCNNModel(self.conv_filters, self.fc_layer_sizes)
        cnn_output = layer(rope_image)
        predictions = Lambda(lambda x: x, name='combined_output')(cnn_output)

        self.model_inputs = [rope_image]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def metadata(self, label_types):
        extra_metadata = {
            'conv_filters': self.conv_filters,
            'fc_layer_sizes': self.fc_layer_sizes,
            'sdf_shape': self.sdf_shape,
        }
        extra_metadata.update(super(OverstretchingCNNModelRunner, self).metadata(label_types))
        return extra_metadata

    def violated(self, observations, sdf_data):
        rope_configurations = observations
        rope_images = link_bot_pycommon.make_rope_images(sdf_data, rope_configurations)
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
        model = OverstretchingCNNModelRunner.load(vars(args), sdf_shape, args.N)
    else:
        model = OverstretchingCNNModelRunner(vars(args), sdf_shape, args.N)

    model.train(train_dataset, validation_dataset, overstretching_label_types, args.epochs, log_path)
    model.evaluate(validation_dataset, overstretching_label_types)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    sdf_shape = dataset.sdf_shape

    model = OverstretchingCNNModelRunner.load(vars(args), sdf_shape, args.N)

    return model.evaluate(dataset, overstretching_label_types)


def main():
    np.set_printoptions(precision=6, suppress=True)
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
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=10)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--n-gpus", type=int, help="number of GPUs", default=1)
    train_subparser.add_argument("--plot", action="store_true")
    train_subparser.add_argument("--val-acc-threshold", type=float, default=None)
    train_subparser.add_argument("--skip-validation", action='store_true')
    train_subparser.add_argument("--early-stopping", action='store_true')
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
