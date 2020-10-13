import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_planning.shooting_method import ShootingMethod
from link_bot_planning.trajectory_optimizer import TrajectoryOptimizer
from link_bot_pycommon.floating_rope_scenario import publish_color_image
from link_bot_pycommon.rviz_animation_controller import RvizSimpleStepper, RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import numpify, remove_batch, add_batch
from sensor_msgs.msg import Image
from state_space_dynamics import model_utils, filter_utils

limit_gpu_mem(10)


def main():
    tf.random.set_seed(0)
    np.random.seed(0)
    colorama.init(autoreset=True)
    np.set_printoptions(linewidth=200, precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'test'], default='val')

    args = parser.parse_args()

    # TODO: REMOVE ME!
    args.mode = 'train'

    rospy.init_node("test_as_inverse_model")

    test_dataset = DynamicsDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)

    # ws = []
    # for i in range(3):
    #     filter_model = filter_utils.load_filter([args.checkpoint])
    #     latent_dynamics_model, _ = model_utils.load_generic_model([args.checkpoint])
    #
    #     w = test_as_inverse_model(filter_model, latent_dynamics_model, test_dataset, test_tf_dataset)
    #     ws.append(w)
    #
    # print(is_close_tf(ws[0], ws[1]))
    # print(is_close_tf(ws[1], ws[2]))
    #

    filter_model = filter_utils.load_filter([args.checkpoint])
    latent_dynamics_model, _ = model_utils.load_generic_model([args.checkpoint])

    test_as_inverse_model(filter_model, latent_dynamics_model, test_dataset, test_tf_dataset)


def test_as_inverse_model(filter_model, latent_dynamics_model, test_dataset, test_tf_dataset):
    scenario = test_dataset.scenario
    shooting_method = ShootingMethod(fwd_model=latent_dynamics_model,
                                     classifier_model=None,
                                     filter_model=filter_model,
                                     scenario=scenario,
                                     params={
                                         'n_samples': 100
                                     })
    trajopt = TrajectoryOptimizer(fwd_model=latent_dynamics_model,
                                  classifier_model=None,
                                  filter_model=filter_model,
                                  scenario=scenario,
                                  params={
                                      "iters": 100,
                                      "length_alpha": 0,
                                      "goal_alpha": 1000,
                                      "constraints_alpha": 0,
                                      "action_alpha": 0,
                                      "initial_learning_rate": 0.0001,
                                  })

    s_color_viz_pub = rospy.Publisher("s_state_color_viz", Image, queue_size=10, latch=True)
    s_next_color_viz_pub = rospy.Publisher("s_next_state_color_viz", Image, queue_size=10, latch=True)
    image_diff_viz_pub = rospy.Publisher("image_diff_viz", Image, queue_size=10, latch=True)

    state = None
    action_horizon = 1
    initial_actions = []
    total_errors = []
    for example_idx, example in enumerate(test_tf_dataset):
        stepper = RvizAnimationController(n_time_steps=test_dataset.steps_per_traj)
        for t in range(test_dataset.steps_per_traj - 1):
            print(example_idx)
            environment = {}
            current_observation = remove_batch(scenario.index_observation_time_batched(add_batch(example), t))
            start_state, _ = filter_model.filter(environment, state, current_observation)

            for j in range(action_horizon):
                left_gripper_position = [0, 0, 0]
                right_gripper_position = [0, 0, 0]
                initial_action = {
                    'left_gripper_position': left_gripper_position,
                    'right_gripper_position': right_gripper_position,
                }
                initial_actions.append(initial_action)
            goal = {
                'rgbd': example['rgbd'][1]
            }
            # actions should just be a single vector with key 'a'
            # actions, planned_path = trajopt.optimize(environment=environment,
            #                                          goal=goal,
            #                                          initial_actions=initial_actions,
            #                                          start_state=start_state)
            true_action = numpify({k: example[k][0] for k in latent_dynamics_model.action_keys})
            actions, planned_path = shooting_method.optimize(current_observation=current_observation,
                                                             environment=environment,
                                                             goal=goal,
                                                             start_state=start_state,
                                                             true_action=true_action)

            for j in range(action_horizon):
                optimized_action = actions[j]
                # optimized_action = {
                #     'left_gripper_position': current_observation['left_gripper'],
                #     'right_gripper_position': current_observation['right_gripper'],
                # }
                true_action = numpify({k: example[k][j] for k in latent_dynamics_model.action_keys})

                # Visualize
                s = numpify(remove_batch(scenario.index_observation_time_batched(add_batch(example), 0)))
                s.update(numpify(remove_batch(scenario.index_observation_features_time_batched(add_batch(example), 0))))
                s_next = numpify(remove_batch(scenario.index_observation_time_batched(add_batch(example), 1)))
                s_next.update(numpify(remove_batch(scenario.index_observation_features_time_batched(add_batch(example), 1))))
                scenario.plot_state_rviz(s, label='t', color="#ff000055", id=1)
                scenario.plot_state_rviz(s_next, label='t+1', color="#aa222255", id=2)
                scenario.plot_action_rviz(s, optimized_action, label='inferred', color='#00ff00', id=1)
                scenario.plot_action_rviz(s, true_action, label='true', color='#0000ff55', id=2)

                publish_color_image(s_color_viz_pub, s['rgbd'][:, :, :3])
                publish_color_image(s_next_color_viz_pub, s_next['rgbd'][:, :, :3])
                diff = s['rgbd'][:, :, :3] - s_next['rgbd'][:, :, :3]
                publish_color_image(image_diff_viz_pub, diff)

                # Metrics
                total_error = 0
                for v1, v2 in zip(optimized_action.values(), true_action.values()):
                    total_error += -np.dot(v1, v2)
                total_errors.append(total_error)

                stepper.step()

        if example_idx > 100:
            break
    print(np.min(total_errors))
    print(np.max(total_errors))
    print(np.mean(total_errors))
    plt.xlabel("total error (meter-ish)")
    plt.hist(total_errors, bins=np.linspace(0, 2, 20))
    plt.show()


if __name__ == '__main__':
    main()
