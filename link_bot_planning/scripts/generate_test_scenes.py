#!/usr/bin/env python
import argparse
import logging
import pathlib
from typing import Optional

import colorama
import numpy as np
import tensorflow as tf

import rosbag
import rospy
from arc_utilities.listener import Listener
from gazebo_msgs.msg import LinkStates
from link_bot_gazebo_python import gazebo_services
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("scenario", type=str)
    parser.add_argument("scenes_dir", type=pathlib.Path)
    parser.add_argument("--n-trials", type=int, default=100)

    args = parser.parse_args()

    rospy.init_node("save_test_scenes")

    return generate_test_scenes(scenario=args.scenario,
                                n_trials=args.n_trials,
                                save_test_scenes_dir=args.scenes_dir)


def generate_test_scenes(scenario: str,
                         n_trials: int,
                         save_test_scenes_dir: Optional[pathlib.Path] = None,
                         ):
    service_provider = gazebo_services.GazeboServices()
    scenario = get_scenario(scenario)

    service_provider.setup_env(verbose=0,
                               real_time_rate=0.0,
                               max_step_size=0.001,
                               play=True)

    scenario.on_before_get_state_or_execute_action()
    scenario.randomization_initialization()

    link_states_listener = Listener("gazebo/link_states", LinkStates)

    env_rng = np.random.RandomState(0)
    action_rng = np.random.RandomState(0)

    params = {
        'repeat_delta_gripper_motion_probability': 0.9,
        'objects_extent':                          [-0.7, 0.7, -0.7, 0.7, 0, 0],
        'objects':                                 [
            'small_box1',
            'small_box2',
            'small_box3',
            'small_box4',
            'small_box5',
            'small_box6',
            'small_box7',
            'small_box8',
            'small_box9',
        ],
        'max_distance_gripper_can_move':           0.15,
        'extent':                                  [-1.2, 1.2, -1.2, 1.2, 0, 0.04],
        'dt':                                      1.0,
        "gripper_action_sample_extent":            [-0.6, 0.6, -0.6, 0.6, 0, 0.04],
    }

    for trial_idx in range(n_trials):
        environment = scenario.get_environment(params)

        scenario.randomize_environment(env_rng, params)

        for i in range(10):
            state = scenario.get_state()
            action = scenario.sample_action(action_rng=action_rng,
                                            environment=environment,
                                            state=state,
                                            action_params=params,
                                            validate=True,
                                            )
            scenario.execute_action(action)

        links_states = link_states_listener.get()
        save_test_scenes_dir.mkdir(exist_ok=True, parents=True)
        bagfile_name = save_test_scenes_dir / f'scene_{trial_idx:04d}.bag'
        rospy.loginfo(f"Saving scene to {bagfile_name}")
        with rosbag.Bag(bagfile_name, 'w') as bag:
            bag.write('links_states', links_states)


if __name__ == '__main__':
    main()
