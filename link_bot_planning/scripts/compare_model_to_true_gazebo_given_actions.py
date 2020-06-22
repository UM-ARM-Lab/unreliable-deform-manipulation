#!/usr/bin/env python

import argparse
import json
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import rospy
import std_srvs
import tensorflow as tf

from link_bot_gazebo import gazebo_services
from link_bot_planning import ompl_viz
from link_bot_classifiers import classifier_utils
from state_space_dynamics import model_utils
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_planning.plan_and_execute import execute_actions
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.ros_pycommon import get_occupancy_data, get_states_dict
from victor import victor_services


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=64 * 64 * 3)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("env_type", choices=['victor', 'gazebo'], default='gazebo', help='victor or gazebo')
    parser.add_argument("params", type=pathlib.Path, help='json file')
    parser.add_argument("actions", type=pathlib.Path, help='csv file of actions')
    parser.add_argument("--outdir", type=pathlib.Path, help="output visualizations here")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--real-time-rate', type=float, default=10.0)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    plt.style.use("paper")

    args = parser.parse_args()

    if args.outdir:
        args.outdir.mkdir(exist_ok=True)

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rospy.init_node('compare_to_true_given_actions')

    # Start Services
    if args.env_type == 'victor':
        service_provider = victor_services.VictorServices()
    else:
        service_provider = gazebo_services.GazeboServices()

    params = json.load(args.params.open('r'))

    scenario = get_scenario(params['scenario'])
    fwd_model, _ = model_utils.load_generic_model(params['fwd_model_dir'])
    classifier_model = classifier_utils.load_generic_model(params['classifier_model_dir'], fwd_model.scenario)

    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=args.real_time_rate,
                               reset_robot=params['reset_robot'],
                               max_step_size=fwd_model.max_step_size)
    service_provider.pause(std_srvs.srv.EmptyRequest())

    full_env_data = get_occupancy_data(env_w_m=fwd_model.full_env_params.w,
                                       env_h_m=fwd_model.full_env_params.h,
                                       res=fwd_model.full_env_params.res,
                                       service_provider=service_provider,
                                       robot_name=scenario.robot_name())

    start_states = get_states_dict(service_provider, fwd_model.state_keys)

    actions = np.genfromtxt(args.actions, delimiter=',')
    T = actions.shape[0]
    actions = actions.reshape([-1, fwd_model.n_action])

    predicted_path = fwd_model.propagate(full_env=full_env_data.data,
                                         full_env_origin=full_env_data.origin,
                                         res=full_env_data.resolution,
                                         start_states=start_states,
                                         actions=actions)
    # Check classifier
    accept_probabilities = []
    for t in range(1, T):
        accept_probability = classifier_model.check_constraint(full_env=full_env_data.data,
                                                               full_env_origin=full_env_data.origin,
                                                               res=full_env_data.resolution,
                                                               states_sequence=predicted_path[:t + 1],
                                                               actions=actions)
        accept_probabilities.append(accept_probability)

    actual_path = execute_actions(service_provider, fwd_model.dt, actions)

    environment = {
        'full_env/env': full_env_data.data,
        'full_env/extent': full_env_data.extent,
    }
    anim = ompl_viz.animate(environment=environment,
                            scenario=scenario,
                            goal=None,
                            accept_probabilities=accept_probabilities,
                            planned_path=predicted_path,
                            actual_path=actual_path)
    if args.outdir:
        outfilename = args.outdir / 'actions_{}.gif'.format(args.actions.stem)
        anim.save(outfilename, writer='imagemagick', dpi=100)
    plt.show()


if __name__ == '__main__':
    main()
