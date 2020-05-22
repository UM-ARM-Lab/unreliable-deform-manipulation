#!/usr/bin/env python
import argparse
import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

import rospy
from link_bot_data.classifier_dataset_utils import compute_label_np
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning import classifier_utils
from state_space_dynamics import model_utils
from link_bot_planning.plan_and_execute import execute_plan
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.ros_pycommon import get_occupancy_data, get_states_dict, xy_move
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_sequences

limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path, nargs='+')
    parser.add_argument("classifier_model_dir", help="classifier", type=pathlib.Path)
    parser.add_argument("test_config", help="json file describing the test", type=pathlib.Path)
    parser.add_argument("--real-time-rate", type=float, default=0.0, help='real time rate')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir)
    classifier_model = classifier_utils.load_generic_model(args.classifier_model_dir, fwd_model.scenario)
    full_env_params = fwd_model.full_env_params

    max_step_size = fwd_model.hparams['dynamics_dataset_hparams']['max_step_size']

    rospy.init_node('test_classifier_from_gazebo')

    test_config = json.load(args.test_config.open("r"))
    actions = np.array(test_config['actions'])

    service_provider = GazeboServices(test_config['object_positions'].keys())
    full_env_data, state = setup(args, service_provider, classifier_model, full_env_params, fwd_model, max_step_size, test_config)
    environment = {
        'full_env/env': full_env_data.data,
        'full_env/origin': full_env_data.origin,
        'full_env/res': full_env_data.resolution,
        'full_env/extent': full_env_data.extent,
    }

    # Prediction
    accept_probabilities, predicted_states_list = predict(actions, classifier_model, full_env_data, fwd_model, environment, state)

    # Execute
    actual_states_list = execute_plan(service_provider, fwd_model.dt, actions)

    # Compute labels
    labeling_params = classifier_model.model_hparams['classifier_dataset_hparams']['labeling_params']
    predicted_states_dict = sequence_of_dicts_to_dict_of_sequences(predicted_states_list)
    actual_states_dict = sequence_of_dicts_to_dict_of_sequences(actual_states_list)
    is_close, _ = compute_label_np(actual_states_dict, labeling_params, predicted_states_dict)
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


def predict(actions, classifier_model, full_env_data, fwd_model, environment, state):
    predicted_states = fwd_model.propagate(full_env=full_env_data.data,
                                           full_env_origin=full_env_data.origin,
                                           res=full_env_data.resolution,
                                           start_states=state,
                                           actions=actions)
    accept_probabilities = classifier_model.check_constraint(environment=environment,
                                                             states_sequence=predicted_states,
                                                             actions=actions)
    return accept_probabilities, predicted_states


def setup(args, service_provider, classifier_model, full_env_params, fwd_model, max_step_size, test_config):
    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=args.real_time_rate,
                               reset_robot=test_config['reset_robot'],
                               max_step_size=max_step_size,
                               stop=True,
                               reset_world=test_config['reset_world'])
    object_moves = {}
    for name, (x, y) in test_config['object_positions'].items():
        object_moves[name] = xy_move(x, y)
    service_provider.move_objects(object_moves=object_moves)
    full_env_data = get_occupancy_data(env_w_m=full_env_params.w,
                                       env_h_m=full_env_params.h,
                                       res=full_env_params.res,
                                       service_provider=service_provider,
                                       robot_name=fwd_model.scenario.robot_name())
    state = get_states_dict(service_provider, fwd_model.states_keys)
    if classifier_model.model_hparams['stdev']:
        state['stdev'] = np.array([0.0], dtype=np.float32)
    return full_env_data, state


if __name__ == '__main__':
    main()
