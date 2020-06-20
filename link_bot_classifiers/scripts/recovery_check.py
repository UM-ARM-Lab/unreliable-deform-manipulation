#!/usr/bin/env python
import argparse
import json
import logging
import pathlib
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_classifiers.analysis_utils import load_models, predict_and_execute, setup
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_pycommon import ros_pycommon
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_sdf_utils import env_from_occupancy_data
from link_bot_pycommon.pycommon import paths_to_json, paths_from_json
from link_bot_pycommon.ros_pycommon import get_states_dict
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import listify, numpify, dict_of_sequences_to_sequence_of_dicts, \
    sequence_of_dicts_to_dict_of_tensors

limit_gpu_mem(4)


def test_params(args):
    rospy.init_node("recovery_check")

    now = time.time()
    basedir = pathlib.Path(f"results/recovery_check/{args.test_params.stem}_{int(now)}")
    basedir.mkdir(exist_ok=True, parents=True)

    test_params = json.load(args.test_params.open("r"))

    # Load and setup
    classifier_model, fwd_model = load_models(args.classifier_model_dir, args.fwd_model_dir)
    service_provider = GazeboServices()
    environment = setup(service_provider, fwd_model, test_params)

    # Teleport the rope to the desired starting state, then get the actual resulting state
    # which could be numerically slightly different, due to small jitter and settling
    desired_start_states = test_params['start_states']
    start_states = []
    for desired_start_state in desired_start_states:
        # TODO:
        # fwd_model.scenario.teleport_to_state(numpify(desired_start_state))
        # fwd_model.scenario.settle()
        start_states.append(fwd_model.scenario.get_state())

    n_start_states = 1
    random_actions = sample_actions(fwd_model.scenario, environment, start_states, args.n_actions_sampled,
                                    args.action_sequence_length)
    start_states = sequence_of_dicts_to_dict_of_tensors(start_states)
    results = predict_and_execute(service_provider, classifier_model, fwd_model, environment, start_states,
                                  random_actions, args.action_sequence_length, args.n_actions_sampled, n_start_states)
    fwd_model, classifier_model, environment, actuals, predictions, accept_probabilities = results

    filename = save_data(args, basedir, environment, actuals, predictions, accept_probabilities, random_actions)

    load_and_compare_predictions_to_actual(filename, no_plot=False)


def load_main(args):
    load_and_compare_predictions_to_actual(args.load_from, args.no_plot)


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
    classifier_model_dir = paths_to_json(args.classifier_model_dir)
    data_and_model_info = {
        'data': listify(data),
        'classifier_model_dir': classifier_model_dir,
        'fwd_model_dir': paths_to_json(args.fwd_model_dir),
    }
    json.dump(data_and_model_info, filename.open("w"))
    return filename


def load_and_compare_predictions_to_actual(load_from: pathlib.Path, no_plot):
    data_and_model_info = json.load(load_from.open("r"))
    saved_data = data_and_model_info['data']
    environment = numpify(saved_data['environment'])
    actuals = [numpify(a_i) for a_i in saved_data['actuals']]
    predictions = [numpify(p_i) for p_i in saved_data['predictions']]
    random_actions = numpify(saved_data['random_actions'])
    accept_probabilities = numpify(saved_data['accept_probabilities'])
    classifier_model_dir = paths_from_json(data_and_model_info['classifier_model_dir'])
    fwd_model_dir = paths_from_json(data_and_model_info['fwd_model_dir'])
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
    if classifier is not None:
        labeling_params = classifier.model_hparams['classifier_dataset_hparams']['labeling_params']
    else:
        labeling_params = {
            'state_key': 'link_bot'
        }
    labeling_params['threshold'] = 0.05
    key = labeling_params['state_key']
    all_predictions_are_far = []
    all_predictions_are_rejected = []
    min_stdevs = []
    median_stdevs = []
    max_stdevs = []
    for i, zipped in enumerate(zip(predictions, actuals, random_actions, accepts_probabilities)):
        prediction, actual, actions, accept_probabilities = zipped
        # [1:] because uncertainty at start is 0
        min_stdev = np.min(prediction['stdev'][1:])
        median_stdev = np.median(prediction['stdev'][1:])
        max_stdev = np.max(prediction['stdev'][1:])
        min_stdevs.append(min_stdev)
        max_stdevs.append(max_stdev)
        median_stdevs.append(median_stdev)
        distance = np.linalg.norm(prediction[key] - actual[key], axis=1)
        print(distance)
        is_close = distance < labeling_params['threshold']
        # [1:] because start state will match perfectly
        last_prediction_is_close = is_close[-1]
        prediction_is_far = np.logical_not(last_prediction_is_close)
        prediction_seq = dict_of_sequences_to_sequence_of_dicts(prediction)
        actual_seq = dict_of_sequences_to_sequence_of_dicts(actual)

        if accepts_probabilities is not None:
            prediction_is_rejected = accept_probabilities[-1] < 0.5
            classifier_says = 'reject' if prediction_is_rejected else 'accept'
            print(f"action sequence {i}, "
                  + f"1-step prediction is close to ground truth? {is_close[1]}, classifier says: {classifier_says}")
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
            anim.save(outfilename, writer='imagemagick', dpi=100)
            plt.close()

    # print(f"mean min stdev {np.mean(min_stdevs):.4f}")
    # print(f"min min stdev {np.min(min_stdevs):.4f}")
    # print(f"mean max stdev {np.mean(max_stdevs):.4f}")
    # print(f"min max stdev {np.min(max_stdevs):.4f}")
    # print(f"mean median stdev {np.mean(median_stdevs):.4f}")
    #
    # if np.all(all_predictions_are_rejected):
    #     print("needs recovery!")


def get_state_and_environment(classifier_model, scenario, service_provider):
    full_env_data = ros_pycommon.get_occupancy_data(env_w_m=classifier_model.full_env_params.w,
                                                    env_h_m=classifier_model.full_env_params.h,
                                                    res=classifier_model.full_env_params.res,
                                                    service_provider=service_provider,
                                                    robot_name=scenario.robot_name())
    environment = env_from_occupancy_data(full_env_data)
    state_dict = get_states_dict(service_provider)
    return environment, state_dict


def sample_actions(scenario, environment, start_states, n_samples, horizon):
    action_rng = np.random.RandomState(0)
    action_sequences = []
    action = None
    for i, start_state in enumerate(start_states):
        action_sequences_for_start_state = []
        for j in range(n_samples):
            action_sequence = []
            for t in range(horizon):
                action = scenario.sample_action(environment=environment,
                                                state=start_state,
                                                last_action=action,
                                                params={},
                                                action_rng=action_rng)
                action_sequence.append(action)
            action_sequence_dict = sequence_of_dicts_to_dict_of_tensors(action_sequence)
            action_sequences_for_start_state.append(action_sequence_dict)
        action_sequence_for_start_state_dict = sequence_of_dicts_to_dict_of_tensors(action_sequences_for_start_state)
        action_sequences.append(action_sequence_for_start_state_dict)
    action_sequences_dict = sequence_of_dicts_to_dict_of_tensors(action_sequences)
    return action_sequences_dict


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()
    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('test_params', help="json file describing the test", type=pathlib.Path)
    generate_parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path, nargs='+')
    generate_parser.add_argument("--classifier-model-dir", help="classifier", type=pathlib.Path)
    generate_parser.add_argument('--n-actions-sampled', type=int, default=10, help='n actions sampled')
    generate_parser.add_argument('--action-sequence-length', type=int, default=3, help='action sequence length')
    generate_parser.set_defaults(func=test_params)
    load_parser = subparsers.add_parser('load')
    load_parser.add_argument('load_from', help="json file with previously generated results", type=pathlib.Path)
    load_parser.add_argument('--no-plot', action='store_true', help='no plot')
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
