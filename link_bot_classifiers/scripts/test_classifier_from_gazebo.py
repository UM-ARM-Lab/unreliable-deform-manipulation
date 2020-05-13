#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_classifiers.visualization import plot_classifier_data, trajectory_plot_from_dataset, trajectory_plot
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning import model_utils, classifier_utils
from link_bot_pycommon.args import my_formatter, point_arg
from link_bot_pycommon.link_bot_pycommon import make_dict_float32
from link_bot_pycommon.ros_pycommon import get_occupancy_data, get_states_dict


def main():
    plt.style.use("slides")
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path, nargs='+')
    parser.add_argument("classifier_model_dir", help="classifier", type=pathlib.Path)
    parser.add_argument("--actions", help="file listing a series of actions", type=pathlib.Path)
    parser.add_argument('--no-plot', action='store_true', help="don't show plots, useful for debugging")
    parser.add_argument('--speed', type=float, default=0.1, help='speed of test actions')
    parser.add_argument("--real-time-rate", type=float, default=0.0, help='real time rate')
    parser.add_argument("--reset-robot", type=point_arg, help='reset robot')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    # use forward model to predict given the input action
    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir)
    classifier_model = classifier_utils.load_generic_model(args.classifier_model_dir, fwd_model.scenario)
    full_env_params = fwd_model.full_env_params

    max_step_size = fwd_model.hparams['dynamics_dataset_hparams']['max_step_size']

    rospy.init_node('test_classifier_from_gazebo')

    service_provider = GazeboServices()
    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=args.real_time_rate,
                               reset_robot=args.reset_robot,
                               max_step_size=max_step_size,
                               stop=False,
                               reset_world=False)

    full_env_data = get_occupancy_data(env_w_m=full_env_params.w,
                                       env_h_m=full_env_params.h,
                                       res=full_env_params.res,
                                       service_provider=service_provider,
                                       robot_name=fwd_model.scenario.robot_name())

    state = get_states_dict(service_provider, fwd_model.states_keys)
    if classifier_model.model_hparams['stdev']:
        state['stdev'] = np.array([0.0], dtype=np.float32)

    if args.actions:
        actions = np.genfromtxt(args.actions, delimiter=',')
        actions_info = []
        for action in actions:
            speed = np.linalg.norm(action)
            theta_deg = np.rad2deg(np.arctan2(action[1], action[0]))
            actions_info.append((speed, theta_deg, np.expand_dims(action, axis=0)))
    else:
        speed = args.speed
        test_inputs = [
            (speed, 135),
            (speed, 90),
            (speed, 45),
            (speed, 180),
            (0, 0),
            (speed, 0),
            (speed, 225),
            (speed, 270),
            (speed, 315),
        ]

        actions_info = []
        for k, (speed, theta_deg) in enumerate(test_inputs):
            theta_rad = np.deg2rad(theta_deg)
            vx = np.cos(theta_rad) * speed
            vy = np.sin(theta_rad) * speed
            action = np.array([[vx, vy], [vx, vy], [vx, vy], [vx, vy]])
            actions_info.append((speed, theta_deg, action))

    fig, axes = plt.subplots(3, 3)
    for k, (speed, theta_deg, action) in enumerate(actions_info):
        pred_states = fwd_model.propagate(full_env=full_env_data.data,
                                          full_env_origin=full_env_data.origin,
                                          res=full_env_data.resolution,
                                          start_states=state,
                                          actions=action)

        environment = {
            'full_env/env': full_env_data.data,
            'full_env/origin': full_env_data.origin,
            'full_env/res': full_env_data.resolution,
            'full_env/extent': full_env_data.extent,
        }
        accept_probability = classifier_model.check_constraint(environement=environment,
                                                               states_sequence=pred_states,
                                                               actions=action)
        accept_probability = float(accept_probability)
        title = 'P(accept) = {:04.3f}%'.format(100 * accept_probability)

        print("v={:04.3f}m/s theta={:04.3f}deg p(accept)={:04.3f}".format(speed, theta_deg, accept_probability))
        if not args.no_plot:
            i, j = np.unravel_index(k, [3, 3])
            trajectory_plot(axes[i, j],
                            fwd_model.scenario,
                            environment=environment,
                            actual_states=None,
                            planned_states=pred_states)
            axes[i, j].set_title(title)

    if not args.no_plot:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()


if __name__ == '__main__':
    main()
