#!/usr/bin/env python
import argparse
import sys

import matplotlib.pyplot as plt
import json
import logging
import pathlib
import time
from typing import Dict, List

import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_classifiers.analysis_utils import predict_and_execute, load_models
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon import ros_pycommon
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_sdf_utils import env_from_occupancy_data
from link_bot_pycommon.pycommon import model_dirs_to_json, model_dirs_from_json
from link_bot_pycommon.ros_pycommon import get_states_dict
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import listify, numpify, dict_of_sequences_to_sequence_of_dicts

limit_gpu_mem(4)


def save_data(args, basedir, environment, actuals, predictions, accept_probabilities, random_actions):
    data = {
        'environment': environment,
        'actuals': actuals,
        'predictions': predictions,
        'accept_probabilities': accept_probabilities,
        'random_actions': random_actions,
    }
    filename = basedir / 'saved_data.json'
    print(Fore.GREEN + f'saving {filename.as_posix()}' + Fore.RESET)
    data_and_model_info = {
        'data': listify(data),
        'classifier_model_dir': model_dirs_to_json(args.classifier_model_dir),
        'fwd_model_dir': model_dirs_to_json(args.fwd_model_dir),
    }
    json.dump(data_and_model_info, filename.open("w"))
    return filename


def test_config(args):
    rospy.init_node("recovery_check")

    now = time.time()
    basedir = pathlib.Path(f"results/recovery_check/{args.test_config.stem}_{int(now)}")
    basedir.mkdir(exist_ok=True)

    test_config = json.load(args.test_config.open("r"))
    random_actions = sample_actions(args.n_actions_sampled, args.action_sequence_length)
    start_configs = [test_config['start_config']] * args.n_actions_sampled
    results = predict_and_execute(args.classifier_model_dir, args.fwd_model_dir, test_config, start_configs, random_actions)
    fwd_model, classifier_model, environment, actuals, predictions, accept_probabilities = results

    filename = save_data(args, basedir, environment, actuals, predictions, accept_probabilities, random_actions)

    load_and_compare_predictions_to_actual(filename, no_plot=False)


def load_main(args):
    load_and_compare_predictions_to_actual(args.load_from, args.no_plot)


def load_and_compare_predictions_to_actual(load_from: pathlib.Path, no_plot):
    data_and_model_info = json.load(load_from.open("r"))
    saved_data = data_and_model_info['data']
    environment = numpify(saved_data['environment'])
    actuals = [numpify(a_i) for a_i in saved_data['actuals']]
    predictions = [numpify(p_i) for p_i in saved_data['predictions']]
    random_actions = numpify(saved_data['random_actions'])
    accept_probabilities = numpify(saved_data['accept_probabilities'])
    classifier_model_dir = model_dirs_from_json(data_and_model_info['classifier_model_dir'])
    fwd_model_dir = model_dirs_from_json(data_and_model_info['fwd_model_dir'])
    classifier_model, fwd_model = load_models(classifier_model_dir=classifier_model_dir, fwd_model_dir=fwd_model_dir)
    basedir = load_from.parent
    compare_predictions_to_actual(basedir,
                                  classifier_model,
                                  environment,
                                  random_actions,
                                  predictions,
                                  actuals,
                                  accept_probabilities,
                                  no_plot)


def compare_predictions_to_actual(basedir: pathlib.Path,
                                  classifier: BaseConstraintChecker,
                                  environment: Dict,
                                  random_actions,
                                  predictions: List,
                                  actuals: List,
                                  accepts_probabilities,
                                  no_plot: bool):
    labeling_params = classifier.model_hparams['classifier_dataset_hparams']['labeling_params']
    labeling_params['threshold'] = 0.05
    key = labeling_params['state_key']
    all_predictions_are_far = []
    all_predictions_are_rejected = []
    min_stdevs = []
    median_stdevs = []
    max_stdevs = []
    max_distance_of_accepted = 0
    for i, zipped in enumerate(zip(predictions, actuals, random_actions, accepts_probabilities)):
        prediction, actual, actions, accept_probabilities = zipped
        # [1:] because uncertainty at start is 0
        min_stdev = np.min(prediction['stdev'][1:])
        median_stdev = np.median(prediction['stdev'][1:])
        max_stdev = np.max(prediction['stdev'][1:])
        min_stdevs.append(min_stdev)
        max_stdevs.append(max_stdev)
        median_stdevs.append(median_stdev)
        is_close = np.linalg.norm(prediction[key] - actual[key], axis=1) < labeling_params['threshold']
        # [1:] because start state will match perfectly
        last_prediction_is_close = is_close[-1]
        prediction_is_far = np.logical_not(last_prediction_is_close)
        prediction_seq = dict_of_sequences_to_sequence_of_dicts(prediction)
        actual_seq = dict_of_sequences_to_sequence_of_dicts(actual)
        distance = classifier.scenario.distance(prediction_seq[0], prediction_seq[-1])

        if is_close[1]:
            max_distance_of_accepted = max(max_distance_of_accepted, distance)

        prediction_is_rejected = accept_probabilities[-1] < 0.5
        classifier_says = 'reject' if prediction_is_rejected else 'accept'
        print(f"action sequence {i}, "
              + f"1-step prediction is close to ground truth {is_close[1]}, classifier says {classifier_says} "
              + f"distance {distance:.3f}")
        all_predictions_are_far.append(prediction_is_far)
        all_predictions_are_rejected.append(prediction_is_rejected)
        if not no_plot:
            anim = classifier.scenario.animate_predictions(environment=environment,
                                                           actions=actions,
                                                           actual=actual_seq,
                                                           predictions=prediction_seq,
                                                           labels=is_close,
                                                           accept_probabilities=accept_probabilities)

            outfilename = basedir / f'action_{i}.gif'
            anim.save(outfilename, writer='imagemagick', dpi=200)
            plt.close()

    print(f"max distance of accepted prediction {max_distance_of_accepted:.3f}")
    print(f"mean min stdev {np.mean(min_stdevs):.4f}")
    print(f"min min stdev {np.min(min_stdevs):.4f}")
    print(f"mean max stdev {np.mean(max_stdevs):.4f}")
    print(f"min max stdev {np.min(max_stdevs):.4f}")
    print(f"mean median stdev {np.mean(median_stdevs):.4f}")

    if max_distance_of_accepted < 0.2:
        print("needs recovery!")


def get_state_and_environment(classifier_model, scenario, service_provider):
    full_env_data = ros_pycommon.get_occupancy_data(env_w_m=classifier_model.full_env_params.w,
                                                    env_h_m=classifier_model.full_env_params.h,
                                                    res=classifier_model.full_env_params.res,
                                                    service_provider=service_provider,
                                                    robot_name=scenario.robot_name())
    environment = env_from_occupancy_data(full_env_data)
    state_dict = get_states_dict(service_provider)
    return environment, state_dict


def sample_actions(n_samples, horizon):
    return tf.random.uniform(shape=[n_samples, horizon, 2], minval=-0.15, maxval=0.15)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()
    test_config_parser = subparsers.add_parser('test_config')
    test_config_parser.add_argument('test_config', help="json file describing the test", type=pathlib.Path)
    test_config_parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path, nargs='+')
    test_config_parser.add_argument("classifier_model_dir", help="classifier", type=pathlib.Path)
    test_config_parser.add_argument('--n-actions-sampled', type=int, default=100)
    test_config_parser.add_argument('--action-sequence-length', type=int, default=6)
    test_config_parser.set_defaults(func=test_config)
    load_parser = subparsers.add_parser('load')
    load_parser.add_argument('load_from', help="json file with previously generated results", type=pathlib.Path)
    load_parser.add_argument('--no-plot', action='store_true')
    load_parser.set_defaults(func=load_main)

    np.set_printoptions(suppress=True, precision=3)
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.get_logger().setLevel(logging.ERROR)

    args = parser.parse_args()
    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
