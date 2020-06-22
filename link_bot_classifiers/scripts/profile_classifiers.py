#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import numpy as np

from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier, DEFAULT_INFLATION_RADIUS
from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_classifiers.nn_classifier import RNNImageClassifierWrapper
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.fishing_3d_scenario import LinkBotScenario
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1.0)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('--n-runs', type=int, default=100)

    args = parser.parse_args()

    scenario = LinkBotScenario()
    rnn_path = pathlib.Path("log_data/rope_6_seq/May_24_11-31-38_617a0bee2a")
    rnn_classifier = RNNImageClassifierWrapper(rnn_path, batch_size=1, scenario=scenario)
    cc_path = pathlib.Path("log_data/collision")
    cc_classifier = CollisionCheckerClassifier(cc_path, inflation_radius=DEFAULT_INFLATION_RADIUS, scenario=scenario)
    none_classifier = NoneClassifier(scenario)

    classifiers = [
        ('rnn', rnn_classifier),
        ('collision checker', cc_classifier),
        ('none', none_classifier),
    ]

    environment = {
        'full_env/env': np.random.randn(100, 100).astype(np.float32),
        'full_env/extent': np.array([-1.0, 1.0, -1.0, 1.0]).astype(np.float32),
        'full_env/res': np.array(0.01).astype(np.float32),
        'full_env/origin': np.array([100, 100]).astype(np.float32),
    }

    n_actions = 4
    action_dim = 2
    state_dim = 22
    n_states = n_actions + 1
    actions = np.random.randn(n_actions, action_dim).astype(np.float32)
    state = {
        'link_bot': np.random.randn(state_dim).astype(np.float32),
        'stdev': np.random.randn(1).astype(np.float32),
    }
    states_sequence = [state] * n_states

    for name, classifier in classifiers:
        # call once first to warm-start
        classifier.check_constraint(environment=environment,
                                    states_sequence=states_sequence,
                                    actions=actions)
        t0 = perf_counter()
        for i in range(args.n_runs):
            classifier.check_constraint(environment=environment,
                                        states_sequence=states_sequence,
                                        actions=actions)
        average_dt = (perf_counter() - t0) / args.n_runs
        print(f'{name:>20s}: {average_dt:.4f}s')


if __name__ == '__main__':
    main()
