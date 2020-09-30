import argparse
import pathlib

import colorama
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
                                  filter_model=filter_model,
                                  scenario=test_dataset.scenario,
                                  params=params)

    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)
    state = None
    action_horizon = 1
    initial_actions = []

    for example in test_tf_dataset:
        for t in range(test_dataset.sequence_length - 1):
            environment = {}
            current_observation = test_dataset.scenario.index_observation_time(example, t)
            start_state, _ = filter_model.filter(environment, state, current_observation)
            for j in range(action_horizon):
                gripper1_position = [0, 0, 0]
                gripper2_position = [0, 0, 0]
                initial_action = {
                    'gripper1_position': gripper1_position,
                    'gripper2_position': gripper2_position,
                }
                initial_actions.append(initial_action)
            goal = {
                'color_depth_image': example['color_depth_image'][1]
            }
            # actions should just be a single vector with key 'a'
            actions, planned_path = trajopt.optimize(environment=environment,
                                                     goal=goal,
                                                     initial_actions=initial_actions,
                                                     start_state=start_state)
            for j in range(action_horizon):
                print(f"j = {j}")
                optimized_action = actions[j]
                true_action = {k: example[k][j] for k in latent_dynamics_model.action_keys}
                print('optimized', optimized_action)
                print('true', true_action)
                total_error = 0
                for v1, v2 in zip(optimized_action.values(), true_action.values()):
                    total_error += tf.linalg.norm(v1 - v2)
                print(total_error)

            return


if __name__ == '__main__':
    main()
