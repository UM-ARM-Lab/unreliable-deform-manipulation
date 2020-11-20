#!/usr/bin/env python
import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.nn_classifier import NNClassifierWrapper
from link_bot_data.visualization import init_viz_env, viz_transition_for_model_t
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.moonshine_utils import repeat, numpify, sequence_of_dicts_to_dict_of_tensors, add_time_dim
from state_space_dynamics import model_utils


def main():
    plt.style.use("slides")
    colorama.init(autoreset=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("saved_state", help="bagfile describing the saved state", type=pathlib.Path)
    parser.add_argument("fwd_model_dir", help="fwd model dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("classifier_model_dir", help="classifier model dir", type=pathlib.Path)
    parser.add_argument("--real-time-rate", type=float, default=0.0, help='real time rate')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    rospy.init_node('test_classifier_from_gazebo')

    fwd_model, _ = model_utils.load_generic_model([pathlib.Path(p) for p in args.fwd_model_dir])
    classifier: NNClassifierWrapper = classifier_utils.load_generic_model([args.classifier_model_dir])

    service_provider = GazeboServices()
    service_provider.setup_env(verbose=0,
                               real_time_rate=0,
                               max_step_size=fwd_model.data_collection_params['max_step_size'])
    service_provider.restore_from_bag(args.saved_state)

    scenario = fwd_model.scenario
    scenario.on_before_get_state_or_execute_action()

    aciton_rng = np.random.RandomState(0)
    # NOTE: perhaps it would make sense to have a "fwd_model" have API for get_env, get_state, sample_action, etc
    #  because the fwd_model knows it's scenario, and importantly it also knows it's data_collection_params
    #  which is what we're using here to pass to the scenario methods
    params = fwd_model.data_collection_params
    environment = numpify(scenario.get_environment(params))
    start_state = numpify(scenario.get_state())

    n_actions = 32
    start_state_tiled = repeat(start_state, n_actions, axis=0, new_axis=True)
    start_states_tiled = add_time_dim(start_state_tiled)
    actions = scenario.sample_action_batch(environment=environment,
                                           state=start_state_tiled,
                                           action_params=params,
                                           action_rng=aciton_rng,
                                           validate=False,
                                           batch_size=n_actions)

    environment_tiled = repeat(environment, n_actions, axis=0, new_axis=True)
    actions_dict = sequence_of_dicts_to_dict_of_tensors(actions)
    actions_dict = add_time_dim(actions_dict)
    predictions, _ = fwd_model.propagate_differentiable_batched(environment=environment_tiled,
                                                                state=start_states_tiled,
                                                                actions=actions_dict)

    # Run classifier
    state_sequence_length = 2
    accept_probabilities, _ = classifier.check_constraint_batched_tf(environment=environment_tiled,
                                                                     predictions=predictions,
                                                                     actions=actions_dict,
                                                                     state_sequence_length=state_sequence_length,
                                                                     batch_size=n_actions)
    # animate over the sampled actions
    anim = RvizAnimation(scenario=scenario,
                         n_time_steps=n_actions,
                         init_funcs=[init_viz_env],
                         t_funcs=[
                             viz_transition_for_model_t({}, fwd_model),
                             ExperimentScenario.plot_accept_probability_t,
                         ],
                         )
    example = {
        'accept_probability': tf.squeeze(accept_probabilities, axis=1),
    }
    example.update(environment)
    example.update(predictions)
    example.update(actions_dict)
    anim.play(example)


if __name__ == '__main__':
    main()
