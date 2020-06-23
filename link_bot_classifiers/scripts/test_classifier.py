#!/usr/bin/env python
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.analysis_utils import predict, execute
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import numpify
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
    parser.add_argument("--real-time-rate", type=float, default=0.0, help='real time rate')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    rospy.init_node('test_classifier_from_gazebo')

    test_config = json.load(args.test_config.open("r"))

    # read actions from config
    actions = test_config['actions']
    n_actions = len(actions)
    actions = [[[numpify(a) for a in actions]]]

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
    predicted_states = predict(fwd_model, environment, start_states, actions, n_actions, 1, 1)

    scenario = fwd_model.scenario
    actual_states_lists = execute(service_provider, scenario, start_states, actions)

    labeling_state_key = classifier.dataset_labeling_params['state_key']

    for actual_states_list, predicted_states_list, actions_list in zip(actual_states_lists, predicted_states, actions):
        for actual_states, predicted_states, actions in zip(actual_states_list, predicted_states_list, actions_list):
            fwd_model.scenario.animate_rviz(environment, actual_states, predicted_states, actions, None, None)


if __name__ == '__main__':
    main()
