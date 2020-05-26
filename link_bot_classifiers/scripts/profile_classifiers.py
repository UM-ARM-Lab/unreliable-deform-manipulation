#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import numpy as np

from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier, DEFAULT_INFLATION_RADIUS
from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_classifiers.rnn_image_classifier import RNNImageClassifierWrapper
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_scenario import LinkBotScenario


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)

    args = parser.parse_args()

    scenario = LinkBotScenario()
    rnn_path = pathlib.Path("log_data/rope_6_seq/May_24_11-31-38_617a0bee2a")
    rnn_classifier = RNNImageClassifierWrapper(rnn_path, batch_size=1, scenario=scenario)
    cc_path = pathlib.Path("log_data/collision")
    cc_classifier = CollisionCheckerClassifier(cc_path, inflation_radius=DEFAULT_INFLATION_RADIUS, scenario=scenario)
    none_classifier = NoneClassifier(scenario)

    classifiers = [
        rnn_classifier,
        cc_classifier,
        none_classifier,
    ]

    environment = {
        'full_env/env': np.random.randn(100, 100),
        'full_env/extent': np.array([-1.0, 1.0, -1.0, 1.0]),
        'full_env/res': np.array(0.01),
        'full_env/origin': np.array([100, 100]),
    }

    n_actions = 4
    action_dim = 2
    state_dim = 20
    n_states = n_actions + 1
    actions = np.random.randn(n_actions, action_dim)
    states_sequence = [{'link_bot': np.random.randn(state_dim)}] * n_states

    for classifier in classifiers:
        t0 = perf_counter()
        classifier.check_constraint(environment=environment,
                                    states_sequence=states_sequence,
                                    actions=actions)
        print('{:.4f}'.format(perf_counter() - t0))


if __name__ == '__main__':
    main()
