import argparse
import pathlib

import colorama
import numpy as np
import tensorflow as tf

from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_planning.trajectory_optimizer import TrajectoryOptimizer
from state_space_dynamics import model_utils, filter_utils


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'test'], default='val')

    args = parser.parse_args()

    test_dataset = DynamicsDataset(args.dataset_dirs)

    params = {
        "iters": 100,
        "length_alpha": 0,
        "goal_alpha": 1000,
        "constraints_alpha": 0,
        "action_alpha": 0,
        "initial_learning_rate": 0.0001,
    }

    filter_model = filter_utils.load_filter([args.checkpoint])
    latent_dynamics_model, _ = model_utils.load_generic_model([args.checkpoint])
    trajopt = TrajectoryOptimizer(fwd_model=latent_dynamics_model,
                                  classifier_model=None,
                                  scenario=test_dataset.scenario,
                                  params=params)

    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)
    state = None
    action_rng = np.random.RandomState(0)
    T = 1
    initial_actions = []

    for example in test_tf_dataset:
        environment = example
        current_observation = example
        start_state, _ = filter_model.filter(environment, state, current_observation)
        for t in range(T):
            initial_action = test_dataset.scenario.sample_action(action_rng=action_rng,
                                                                 environment=environment,
                                                                 state=start_state,
                                                                 data_collection_params=test_dataset.data_collection_params,
                                                                 action_params=test_dataset.data_collection_params)
            initial_actions.append(initial_action)
        goal = {
            'goal_obs': example['color_depth_image'][1]
        }
        actions, planned_path = trajopt.optimize(environment=environment,
                                                 goal=goal,
                                                 initial_actions=initial_actions,
                                                 start_state=start_state)
        optimized_action = actions[0]
        true_action = {k: example[k] for k in latent_dynamics_model.action_keys}
        print('optimized', optimized_action)
        print('true', true_action)
        total_error = 0
        for v1, v2 in zip(optimized_action.values(), true_action.values()):
            total_error += tf.linalg.norm(v1 - v2)
        print(total_error)


if __name__ == '__main__':
    main()
