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

import rospy
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.analysis_utils import predict_and_execute
from link_bot_planning.plan_and_execute import execute_plan
from link_bot_pycommon import ros_pycommon
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import env_from_occupancy_data
from link_bot_pycommon.ros_pycommon import get_states_dict
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, sequence_of_dicts_to_dict_of_np_arrays
from state_space_dynamics import model_utils
from std_srvs.srv import EmptyRequest


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("test_config", help="json file describing the test", type=pathlib.Path)

    fwd_model_dirs = [pathlib.Path(f"./ss_log_dir/tf2_rope/{i}") for i in range(8)]
    classifier_model_dir = pathlib.Path('log_data/rope_2_seq/May_24_01-12-08_617a0bee2a')

    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=3)
    np.random.seed(0)
    tf.random.set_seed(0)
    rospy.init_node("recovery_check")
    tf.get_logger().setLevel(logging.ERROR)

    test_config = json.load(args.test_config.open("r"))

    n_actions_sampled = 10
    action_sequence_length = 5
    random_actions = sample_actions(n_actions_sampled, action_sequence_length)

    # default_joint_angles = [120, -120, 120, -120, 120, -120, 120, -120, 120, -120]
    # default_joint_angles = [60, 30, 30, 40, 50, 70, 30, 80, 110, 0]

    start_configs = []
    results = predict_and_execute(classifier_model_dir, fwd_model_dirs, test_config, start_configs, random_actions)
    fwd_model, classifier_model, environment, actuals, predictions, accept_probabilities = results

    labeling_params = {
        'threshold': 0.08,
        'state_key': 'link_bot',
    }

    now = time.time()
    basedir = pathlib.Path(f"results/recovery_check/{int(now)}")
    basedir.mkdir(exist_ok=True)
    compare_predictions_to_actual(basedir=basedir,
                                  scenario=classifier_model.scenario,
                                  environment=environment,
                                  random_actions=random_actions,
                                  predictions=predictions,
                                  actuals=actuals,
                                  labeling_params=labeling_params)


def execute(args, service_provider, dt, random_actions):
    actual_state_sequences = []
    for actions in random_actions:
        set_rope_config(args, service_provider)
        service_provider.stop_robot(EmptyRequest())
        actual_state_sequence = execute_plan(service_provider, dt, actions)
        actual_state_sequences.append(actual_state_sequence)
    return actual_state_sequences


def predict(fwd_models, n_actions_sampled, random_actions, state_dict):
    state_vec = state_dict['link_bot']
    state_batched = tf.expand_dims(tf.stack([state_vec] * n_actions_sampled, axis=0), axis=1)
    state_dict_batched = {
        'link_bot': state_batched,
    }
    predictions = fwd_models.propagate_differentiable_batched(start_states=state_dict_batched,
                                                              actions=random_actions)
    return dict_of_sequences_to_sequence_of_dicts(predictions)


def compare_predictions_to_actual(basedir: pathlib.Path,
                                  scenario: ExperimentScenario,
                                  environment: Dict,
                                  random_actions,
                                  predictions: List,
                                  actuals: List,
                                  labeling_params: Dict):
    key = labeling_params['state_key']
    all_predictions_are_far = []
    for i, (prediction, actual_seq, actions) in enumerate(zip(predictions, actuals, random_actions)):
        prediction_seq = dict_of_sequences_to_sequence_of_dicts(prediction)
        actual = sequence_of_dicts_to_dict_of_np_arrays(actual_seq)
        all_prediction_is_close = np.linalg.norm(prediction[key] - actual[key], axis=1) < labeling_params['threshold']
        # [1:] because start state will match perfectly
        last_prediction_is_close = all_prediction_is_close[-1]
        print(f"action sequence {i}, final prediction is close to ground truth {last_prediction_is_close}")
        prediction_is_far = np.logical_not(last_prediction_is_close)
        all_predictions_are_far.append(prediction_is_far)
        anim = scenario.animate_predictions(environment=environment,
                                            actions=actions,
                                            actual=actual_seq,
                                            predictions=prediction_seq,
                                            labels=all_prediction_is_close)
        outfilename = basedir / f'action_{i}.gif'
        anim.save(outfilename, writer='imagemagick', dpi=200)
        plt.close()

    if np.all(all_predictions_are_far):
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


def load_models(args, fwd_model_dirs, classifier_model_dir, scenario):
    fwd_models, _ = model_utils.load_generic_model(fwd_model_dirs)
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, scenario=scenario)
    return fwd_models, classifier_model


def sample_actions(n_samples, horizon):
    return tf.random.uniform(shape=[n_samples, horizon, 2], minval=-0.15, maxval=0.15)


if __name__ == '__main__':
    main()
