#!/usr/bin/env python
import argparse
import json
import pathlib
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_classifiers.analysis_utils import predict_and_execute
from link_bot_data.classifier_dataset_utils import compute_label_np
from link_bot_pycommon.args import my_formatter
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts

limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path, nargs='+')
    parser.add_argument("classifier_model_dir", help="classifier", type=pathlib.Path)
    parser.add_argument("test_config", help="json file describing the test", type=pathlib.Path)
    parser.add_argument("--labeling-params",
                        help="use labeling params, if not provided we use what the classifier was trained with",
                        type=pathlib.Path)
    parser.add_argument("--real-time-rate", type=float, default=0.0, help='real time rate')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    rospy.init_node('test_classifier_from_gazebo')

    test_config = json.load(args.test_config.open("r"))

    # read actions from config
    actions = np.expand_dims(np.array(test_config['actions']), axis=0).astype(np.float32)

    fwd_model_dir = args.fwd_model_dir
    classifier_model_dir = args.classifier_model_dir

    start_configs = [test_config['start_config']]
    results = predict_and_execute(classifier_model_dir, fwd_model_dir, test_config, start_configs, actions)
    fwd_model, classifier_model, environment, actuals, predictions, accept_probabilities = results
    actual = actuals[0]
    actions = actions[0]
    prediction = predictions[0]
    accept_probabilities = accept_probabilities[0]

    visualize(accept_probabilities, actions, actual, args, classifier_model, environment, fwd_model, prediction)


def visualize(accept_probabilities,
              actions,
              actual_states_dict: Dict,
              args,
              classifier_model,
              environment,
              fwd_model,
              predicted_states_dict: Dict):
    if args.labeling_params is None:
        labeling_params = classifier_model.model_hparams['classifier_dataset_hparams']['labeling_params']
    else:
        labeling_params = json.load(args.labeling_params.open("r"))
    is_close = compute_label_np(actual_states_dict, predicted_states_dict, labeling_params)
    actual_states_list = dict_of_sequences_to_sequence_of_dicts(actual_states_dict)
    predicted_states_list = dict_of_sequences_to_sequence_of_dicts(predicted_states_dict)
    is_close = is_close.astype(np.float32)

    anim = fwd_model.scenario.animate_predictions(environment=environment,
                                                  actions=actions,
                                                  actual=actual_states_list,
                                                  predictions=predicted_states_list,
                                                  labels=is_close,
                                                  accept_probabilities=accept_probabilities)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    if args.save:
        now = int(time.time())
        outdir = pathlib.Path("results") / 'test_classifier' / f"{now}"
        outdir.mkdir(parents=True)
        filename = outdir / f"{args.test_config.stem}.gif"
        print(f"saving {filename}")
        anim.save(filename, writer='imagemagick', dpi=200, fps=1)


if __name__ == '__main__':
    main()
