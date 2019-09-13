#!/usr/bin/env python
from __future__ import print_function

import keras
import numpy as np
import tensorflow as tf

from link_bot_classifiers import base_classifier_runner
from link_bot_data.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_classifiers.sdf_cnn_model_runner import SDFCNNModelRunner
from link_bot_pycommon import experiments_util


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = SDFCNNModelRunner.load(args.checkpoint)
    else:
        args_dict = {
            'sdf_shape': sdf_shape,
            'beta': 1e-2,
            'conv_filters': [
                (32, (5, 5)),
                (32, (5, 5)),
                (16, (3, 3)),
                (16, (3, 3)),
            ],
            'cnn_fc_layer_sizes': [256, 256],
            'fc_layer_sizes': [32, 32],
            'sigmoid_scale': 100,
            'N': train_dataset.N
        }
        args_dict.update(base_classifier_runner.make_args_dict(args))
        model = SDFCNNModelRunner(args_dict)

    model.train(train_dataset, validation_dataset, args.label_types_map, log_path, args)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    model = SDFCNNModelRunner.load(args.checkpoint)

    names = [weight.name for layer in model.keras_model.layers for weight in layer.weights]
    weights = model.keras_model.get_weights()
    sdf_weight_names = [weight.name for layer in model.sdf_input_model.layers for weight in layer.weights]

    sdf_weights = []
    for name, weight in zip(names, weights):
        print(name, weight)
        if name in sdf_weight_names:
            sdf_weights.append(weight)

    model.sdf_input_model.set_weights(sdf_weights)

    # view the activations of the SDF function model head
    sdf_output_model = keras.models.Model(inputs=model.model_inputs, outputs=model.sdf_function_prediction)
    sdf_output_model.set_weights(sdf_weights)

    model_output_names = [output.op.name.split("/")[0] for output in model.keras_model.outputs]
    generator = dataset.generator_for_labels(model_output_names, args.label_types_map, args.batch_size)
    x, _ = generator[0]
    predicted_point = model.sdf_input_model.predict(x)
    sdf_function_prediction = sdf_output_model.predict(x)
    rope = x['rope_configuration']
    print(np.hstack((rope, predicted_point, sdf_function_prediction)))

    model.evaluate(dataset, args.label_types_map)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_classifier_runner.base_parser()

    train_subparser.set_defaults(func=train)
    eval_subparser.set_defaults(func=evaluate)
    show_subparser.set_defaults(func=SDFCNNModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
