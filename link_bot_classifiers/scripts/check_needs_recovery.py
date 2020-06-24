#!/usr/bin/env python
import argparse
import json
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.analysis_utils import execute, predict_and_classify
from link_bot_data.classifier_dataset_utils import compute_is_close_tf
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors
from state_space_dynamics import model_utils

limit_gpu_mem(9)


def main():
    plt.style.use("slides")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("test_config", help="json file describing the test", type=pathlib.Path)
    parser.add_argument("--labeling-params",
                        help="use labeling params, if not provided we use what the classifier was trained with",
                        type=pathlib.Path)
    parser.add_argument("--n-action-sequences", type=int, default=10, help='n action sequences')
    parser.add_argument("--action-sequence-length", type=int, default=10, help='action sequence length')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    rospy.init_node('test_classifier_from_gazebo')

    test_config = json.load(args.test_config.open("r"))

    fwd_model, _ = model_utils.load_generic_model([pathlib.Path(p) for p in test_config['fwd_model_dirs']])
    classifier = classifier_utils.load_generic_model(test_config['classifier_model_dir'], fwd_model.scenario)

    service_provider = GazeboServices()
    service_provider.setup_env(verbose=0, real_time_rate=0, max_step_size=fwd_model.data_collection_params['max_step_size'])
    environment = get_environment_for_extents_3d(extent=test_config['extent'],
                                                 res=fwd_model.data_collection_params['res'],
                                                 service_provider=service_provider,
                                                 robot_name=fwd_model.scenario.robot_name())
    start_state = fwd_model.scenario.get_state()
    start_state = make_dict_tf_float32(start_state)
    start_states = [start_state]

    # read actions from config
    action_sequences = sample_actions(args, fwd_model.scenario, environment, start_state, fwd_model.data_collection_params)

    predicted_states, accept_probabilities = predict_and_classify(fwd_model=fwd_model,
                                                                  classifier=classifier,
                                                                  environment=environment,
                                                                  start_states=start_states,
                                                                  actions=[action_sequences],
                                                                  n_actions=args.action_sequence_length,
                                                                  n_start_states=1,
                                                                  n_actions_sampled=args.n_action_sequences)

    scenario = fwd_model.scenario
    actual_states_lists = execute(service_provider, scenario, start_states, action_sequences)

    for i in args.n_action_sequences:
        actual_states = actual_states_lists[0][i]
        predicted_states = predicted_states[0][i]
        actions = action_sequences[0][i]
        accept_probabilities = accept_probabilities[0][i]
        actual_states_dict = sequence_of_dicts_to_dict_of_tensors(actual_states)
        predicted_states_dict = sequence_of_dicts_to_dict_of_tensors(predicted_states)
        labels = compute_is_close_tf(actual_states_dict, predicted_states_dict, classifier.dataset_labeling_params)
        fwd_model.scenario.animate_rviz(environment, actual_states, predicted_states, actions, labels, accept_probabilities)


def sample_actions(args, scenario: Base3DScenario, environment: Dict, start_state: Dict, params: Dict):
    action_sequences = []
    rng = np.random.RandomState()

    for i in range(args.n_action_sequences):
        action_sequence = []
        for t in range(args.action_sequence_length):
            action = scenario.sample_action(environment=environment,
                                            state=start_state,
                                            params=params,
                                            action_rng=rng)
            action_sequence.append(action)
        action_sequences.append(action_sequence)
    return action_sequences


if __name__ == '__main__':
    main()
