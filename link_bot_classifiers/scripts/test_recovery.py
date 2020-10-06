#!/usr/bin/env python
import argparse
import gzip
import json
import pathlib

import colorama
import numpy as np

import rospy
from link_bot_classifiers.recovery_policy_utils import load_generic_model
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("infile", help="json.gz file describing the test cases", type=pathlib.Path)
    parser.add_argument("recovery_model_dir", help="recovery model dir", type=pathlib.Path)

    args = parser.parse_args()

    rospy.init_node('test_recovery')

    scenario = FloatingRopeScenario()

    with gzip.open(args.infile, 'rb') as data_file:
        data_str = data_file.read()
    data = json.loads(data_str.decode("utf-8"))

    n_time_steps = len(data)
    time_steps = np.arange(n_time_steps)
    print(f"{n_time_steps} examples")
    rng = np.random.RandomState(0)
    for example in data:
        state = example['state']
        environment = example['environment']

        recovery_policy = load_generic_model(args.recovery_model_dir, scenario, rng)
        recovery_action = recovery_policy(environment=environment, state=state)

        # scenario.plot_environment_rviz(environment)
        # scenario.plot_state_rviz(state, label='state', color='red')
        # scenario.plot_action_rviz(state, recovery_action, label='proposed', color='magenta')
        input("press enter to see the next example")
    print("done.")


if __name__ == '__main__':
    main()
