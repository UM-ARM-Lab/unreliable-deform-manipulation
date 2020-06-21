#!/usr/bin/env python
import argparse
import json
import logging
import pathlib
import time

import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_classifiers.analysis_utils import predict_and_execute, setup, sample_actions
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.pycommon import paths_to_json, paths_from_json
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import listify, numpify
from state_space_dynamics import model_utils

limit_gpu_mem(4)


def test_params(args):
    rospy.init_node("recovery_check")

    now = time.time()
    basedir = pathlib.Path(f"results/recovery_check/{args.test_params.stem}_{int(now)}")
    basedir.mkdir(exist_ok=True, parents=True)

    test_params = json.load(args.test_params.open("r"))

    # Load and setup
    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir)
    scenario = fwd_model.scenario
    service_provider = GazeboServices()
    environment = setup(service_provider, fwd_model, test_params)

    start_states = [scenario.get_state()]
    n_start_states = len(start_states)

    random_actions = sample_actions(scenario, environment, start_states, args.n_actions_sampled, args.action_sequence_length)
    results = predict_and_execute(service_provider, fwd_model, environment, start_states, random_actions,
                                  args.action_sequence_length, args.n_actions_sampled, n_start_states)
    fwd_model, environment, actuals, predictions = results

    filename = save_data(args, basedir, environment, actuals, predictions, random_actions)

    load_and_compare_predictions_to_actual(filename, no_plot=False)


def load_main(args):
    rospy.init_node("recovery_check")
    load_and_compare_predictions_to_actual(args.load_from, args.no_plot)


def save_data(args, basedir, environment, actuals, predictions, actions):
    data = {
        'environment': environment,
        'actuals': actuals,
        'predictions': predictions,
        'actions': actions,
    }
    filename = basedir / 'saved_data.json'
    print(Fore.GREEN + f'saving {filename.as_posix()}' + Fore.RESET)
    data_and_model_info = {
        'data': listify(data),
        'fwd_model_dir': paths_to_json(args.fwd_model_dir),
    }
    json.dump(data_and_model_info, filename.open("w"))
    return filename


def load_and_compare_predictions_to_actual(load_from: pathlib.Path, no_plot):
    data_and_model_info = json.load(load_from.open("r"))
    saved_data = data_and_model_info['data']
    environment = numpify(saved_data['environment'])
    # the first dimension is number of start states, second dimension is number of random actions
    actuals = [[numpify(a_ij) for a_ij in a_i] for a_i in saved_data['actuals']]
    predictions = [[numpify(p_ij) for p_ij in p_i] for p_i in saved_data['predictions']]
    action_sequences = [[numpify(a_ij) for a_ij in a_i] for a_i in saved_data['actions']]
    # here we assume first dim is 0
    actuals = actuals[0]
    action_sequences = action_sequences[0]
    predictions = predictions[0]

    fwd_model_dir = paths_from_json(data_and_model_info['fwd_model_dir'])
    fwd_model, _ = model_utils.load_generic_model(fwd_model_dir)
    scenario = fwd_model.scenario

    basedir = load_from.parent

    # FIXME: get these from a file
    labeling_params = {
        'state_key': 'link_bot',
        'threshold': 0.0853
    }

    key = labeling_params['state_key']
    min_dist = np.finfo(np.float32).max
    for i, (prediction, actual, actions) in enumerate(zip(predictions, actuals, action_sequences)):
        scenario.plot_environment_rviz(environment)

        n_states = len(actual)
        time_steps = np.arange(n_states)
        anim = RvizAnimationController(time_steps)

        for t in range(n_states):
            # while not anim.done:
            #     t = anim.t()
            actual_t = numpify(actual[t])
            pred_t = numpify(prediction[t])
            scenario.plot_state_rviz(actual_t, label='actual', color='#ff0000aa')
            scenario.plot_state_rviz(pred_t, label='pred', color='#0000ffaa')

            if t > 0:
                distance = np.linalg.norm(pred_t[key] - actual_t[key])
                min_dist = min(min_dist, distance)

            # this will return after some amount of time either because the animation is "playing" or because the user stepped forward
            # anim.step()

    print(min_dist)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()
    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('test_params', help="json file describing the test", type=pathlib.Path)
    generate_parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path, nargs='+')
    generate_parser.add_argument('--n-actions-sampled', type=int, default=25, help='n actions sampled')
    generate_parser.add_argument('--action-sequence-length', type=int, default=1, help='action sequence length')
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
