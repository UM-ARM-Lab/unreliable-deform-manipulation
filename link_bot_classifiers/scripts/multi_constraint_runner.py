#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import tensorflow as tf
from colorama import Style

from link_bot_classifiers import base_classifier_runner
from link_bot_classifiers.multi_constraint_model_runner import MultiConstraintModelRunner
from link_bot_data.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util


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
            'beta': 1e-2,
            'fc_layer_sizes': [32, 32],
            'sigmoid_scale': 100,
            'N': train_dataset.N,
        }
        args_dict.update(base_classifier_runner.make_args_dict(args))
        model = MultiConstraintModelRunner(args_dict)

    model.train(train_dataset, validation_dataset, args.label_types_map, log_path, args)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    model = MultiConstraintModelRunner.load(args.checkpoint)

    # normal evaluation
    # print(Style.BRIGHT + "Combined Model:" + Style.NORMAL)
    # model.evaluate(dataset, args.label_types_map)

    sdf_only_true_positive = 0
    sdf_only_true_negative = 0
    ovs_only_true_positive = 0
    sdf_and_ovs_true_positive = 0
    neither_true_negative = 0
    either_true_positive = 0
    either_true_negative = 0
    sdf_only_total = 0
    ovs_only_total = 0
    sdf_and_ovs_total = 0
    neither_total = 0
    either_pos_total = 0
    either_neg_total = 0
    sdf_only_negative_total = 1
    total = 0
    sdf_correct = 0
    ovs_correct = 0
    correct = 0
    _correct = 0
    xx = 0
    generator = dataset.generator(args.batch_size)
    for batch_i in range(len(generator)):
        x, y_true = generator[batch_i]
        combined_predictions, sdf_predictions, ovs_predictions = model.keras_model.predict(x)
        true_sdf = y_true['SDF']
        true_ovs = y_true['Overstretching']
        true_combined = y_true['combined']

        for yi_sdf, yi_ovs, yi_combined, yi_hat_sdf, yi_hat_ovs, yi_hat_combined in zip(true_sdf, true_ovs, true_combined,
                                                                                        sdf_predictions, ovs_predictions,
                                                                                        combined_predictions):
            yi_hat_sdf = np.round(yi_hat_sdf).astype(np.int)
            yi_hat_ovs = np.round(yi_hat_ovs).astype(np.int)
            yi_hat_combined = (yi_hat_combined > 0.9999).astype(np.int)

            if yi_sdf == yi_hat_sdf:
                sdf_correct += 1
            if yi_ovs == yi_hat_ovs:
                ovs_correct += 1
            if (yi_sdf or yi_ovs) == (yi_hat_sdf or yi_hat_ovs):
                _correct += 1

            if yi_sdf == 0:
                if yi_hat_sdf == 0:
                    sdf_only_true_negative += 1
                sdf_only_negative_total += 1

            if yi_sdf and not yi_ovs:
                if yi_hat_sdf and not yi_hat_ovs:
                    sdf_only_true_positive += 1
                sdf_only_total += 1
            elif not yi_sdf and yi_ovs:
                if not yi_hat_sdf and yi_hat_ovs:
                    ovs_only_true_positive += 1
                ovs_only_total += 1
            elif yi_sdf and yi_ovs:
                if yi_hat_sdf and yi_hat_ovs:
                    sdf_and_ovs_true_positive += 1
                sdf_and_ovs_total += 1
            else:
                if not yi_hat_sdf and yi_hat_ovs:
                    neither_true_negative += 1
                neither_total += 1

            if yi_combined:
                if yi_hat_combined:
                    either_true_positive += 1
                either_pos_total += 1
            elif not yi_combined:
                if not yi_hat_combined:
                    either_true_negative += 1
                either_neg_total += 1
            if yi_combined.astype(np.int) == yi_hat_combined:
                correct += 1
            total += 1

    print(Style.BRIGHT + "Custom Metrics:" + Style.NORMAL)
    print("SDF Accuracy: {}/{}".format(sdf_correct, total))
    print("Overstretching Accuracy: {}/{}".format(ovs_correct, total))
    print("SDF Only True Positive: {}/{}".format(sdf_only_true_positive, sdf_only_total))
    print("SDF Only True Negativ : {}/{}".format(sdf_only_true_negative, sdf_only_negative_total))
    print("Overstretching Only True Positive: {}/{}".format(ovs_only_true_positive, ovs_only_total))
    print("Both True Positive: {}/{}".format(sdf_and_ovs_true_positive, sdf_and_ovs_total))
    print("Niether True Negative: {}/{}".format(neither_true_negative, neither_total))
    print("Either True Postive: {}/{}".format(either_true_positive, either_pos_total))
    print("Either True Negative: {}/{}".format(either_true_negative, either_neg_total))
    print("Combined Accuracy: {}/{}".format(correct, total))
    print("Combined Accuracy*: {}/{}".format(_correct, total))


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_classifier_runner.base_parser()

    train_subparser.set_defaults(func=train)
    # eval_subparser.set_defaults(func=evaluate)
    eval_subparser.set_defaults(func=MultiConstraintModelRunner.evaluate_main)
    show_subparser.set_defaults(func=MultiConstraintModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
