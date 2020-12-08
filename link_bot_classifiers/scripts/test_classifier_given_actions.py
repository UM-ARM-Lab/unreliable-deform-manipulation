#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import hjson
import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_classifiers.test_classifier import test_classifier
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.experiment_scenario import ExperimentScenario


def main():
    plt.style.use("slides")
    colorama.init(autoreset=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="fwd model dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("classifier_model_dir", help="classifier model dir", type=pathlib.Path)
    parser.add_argument("actions", help="actions hjson file", type=pathlib.Path)
    parser.add_argument("--saved-state", help="bagfile describing the saved state", type=pathlib.Path)

    args = parser.parse_args()

    rospy.init_node('test_classifier_from_gazebo')

    with args.actions.open("r") as actions_file:
        actions = hjson.load(actions_file)
    n_actions = len(actions)

    def _generate_actions(environment: Dict,
                          start_state_tiled: Dict,
                          scenario: ExperimentScenario,
                          params: Dict,
                          n_actions: int):
        return actions

    test_classifier(classifier_model_dir=args.classifier_model_dir,
                    fwd_model_dir=args.fwd_model_dir,
                    saved_state=args.saved_state,
                    generate_actions=_generate_actions,
                    n_actions=n_actions)


if __name__ == '__main__':
    main()
