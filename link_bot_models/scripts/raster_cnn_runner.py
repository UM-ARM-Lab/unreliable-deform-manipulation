#!/usr/bin/env python
from __future__ import print_function

import json

import numpy as np
import tensorflow as tf
from keras.layers import Input, Concatenate, Dense
from keras.models import Model

from link_bot_models import base_model_runner
from link_bot_models.base_model_runner import BaseModelRunner
from link_bot_models.components.raster_points_layer import RasterPoints
from link_bot_models.components.simple_cnn_layer import simple_cnn_layer
from link_bot_pycommon import experiments_util
from video_prediction import datasets


class RasterCNNModelRunner(BaseModelRunner):

    def __init__(self, args_dict, inputs):
        super(RasterCNNModelRunner, self).__init__(args_dict)
        self.sdf_shape = args_dict['sdf_shape']
        self.conv_filters = args_dict['conv_filters']
        self.fc_layer_sizes = args_dict['fc_layer_sizes']
        sdf_shape = inputs['sdf'].shape[1:3].as_list()

        sdf = Input(tensor=inputs['sdf'])
        rope_config = Input(tensor=inputs['rope_configurations'])
        sdf_resolution = Input(tensor=inputs['sdf_resolution'])
        sdf_origin = Input(tensor=inputs['sdf_origin'])

        rope_image = RasterPoints(sdf_shape)([rope_config, sdf_resolution, sdf_origin])
        combined_image = Concatenate()([sdf, rope_image])

        cnn_output = simple_cnn_layer(self.conv_filters, self.fc_layer_sizes)(combined_image)
        predictions = Dense(1, activation='sigmoid', name='combined_output')(cnn_output)

        self.model_inputs = [sdf, rope_config, sdf_resolution, sdf_origin]
        self.keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        self.keras_model.compile(optimizer='adam',
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'],
                                 target_tensors=[inputs['constraints']])

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
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    dataset_hparams_dict = json.load(open(args.dataset_hparams_dict, 'r'))

    VideoDataset = datasets.get_dataset_class(args.dataset)
    train_dataset = VideoDataset(args.input_dir,
                                 mode='train',
                                 num_epochs=args.epochs,
                                 seed=args.seed,
                                 hparams_dict=dataset_hparams_dict,
                                 hparams=args.dataset_hparams)
    print(train_dataset.num_examples_per_epoch())
    val_dataset = VideoDataset(args.input_dir,
                               mode='val',
                               num_epochs=args.epochs,
                               seed=args.seed,
                               hparams_dict=dataset_hparams_dict,
                               hparams=args.dataset_hparams)

    batch_size = args.batch_size
    train_tf_dataset = train_dataset.make_dataset(batch_size)
    train_iterator = train_tf_dataset.make_one_shot_iterator()
    train_handle = train_iterator.string_handle()
    train_iterator = tf.data.Iterator.from_string_handle(train_handle, train_tf_dataset.output_types,
                                                         train_tf_dataset.output_shapes)
    train_inputs = train_iterator.get_next()

    val_tf_dataset = val_dataset.make_dataset(batch_size)
    val_iterator = val_tf_dataset.make_one_shot_iterator()
    val_handle = val_iterator.string_handle()
    val_iterator = tf.data.Iterator.from_string_handle(val_handle, val_tf_dataset.output_types,
                                                       val_tf_dataset.output_shapes)
    val_inputs = val_iterator.get_next()

    # Now that we have the input tensors, so we can construct our Keras model
    if args.checkpoint:
        raise NotImplementedError()
        # model = RasterCNNModelRunner.load(args.checkpoint)
    else:
        args_dict = {
            'sdf_shape': train_dataset.hparams.sdf_shape,
            'conv_filters': [
                (32, (5, 5)),
                (32, (5, 5)),
                (16, (3, 3)),
                (16, (3, 3)),
            ],
            'fc_layer_sizes': [256, 256],
            'N': train_dataset.hparams.rope_config_dim,
        }
        args_dict.update(base_model_runner.make_args_dict(args))
        train_model = RasterCNNModelRunner(args_dict, train_inputs)
        val_model = RasterCNNModelRunner(args_dict, val_inputs)

    train_model.train(train_dataset, log_path, args)


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_model_runner.base_parser()

    train_subparser.set_defaults(func=train)
    eval_subparser.set_defaults(func=RasterCNNModelRunner.evaluate_main)
    show_subparser.set_defaults(func=RasterCNNModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
