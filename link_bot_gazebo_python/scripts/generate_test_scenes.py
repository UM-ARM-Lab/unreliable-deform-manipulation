import argparse
import pathlib
import time

import colorama
import hjson
import numpy as np
import tensorflow as tf

import rosbag
import rospy
from arc_utilities.ros_helpers import Listener
from gazebo_msgs.msg import LinkStates
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("scenario_name", type=str)
    parser.add_argument("data_collection_params", type=pathlib.Path)
    parser.add_argument("n_trials", type=int)
    parser.add_argument("nickname", type=str)

    args = parser.parse_args()

    rospy.init_node("generate_test_scenarios")

    scenario = get_scenario(args.scenario_name)
    scenario.randomization_initialization()

    link_states_listener = Listener("gazebo/link_states", LinkStates)

    with args.data_collection_params.open("r") as data_collection_params_file:
        data_collection_params = hjson.load(data_collection_params_file)

    unique_nickname = f"{args.nickname}_{int(time.time())}"
    bagfiledir = pathlib.Path("test_scenes") / unique_nickname
    print(bagfiledir)
    bagfiledir.mkdir(parents=True)
    for trial_idx in range(args.n_trials):
        print(trial_idx)
        bagfilename = bagfiledir / f'scene_{trial_idx:04d}.bag'
        with rosbag.Bag(bagfilename, 'w') as bag:
            try:
                np.random.seed(trial_idx)
                tf.random.set_seed(trial_idx)
                env_rng = np.random.RandomState(trial_idx)

                scenario.randomize_environment(env_rng, data_collection_params)

                links_states = link_states_listener.get()

                bag.write('links_states', links_states)
            finally:
                bag.close()


if __name__ == '__main__':
    main()
