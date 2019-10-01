#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import rospy
import tensorflow as tf
from colorama import Fore

from link_bot_data import random_environment_data_utils
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import get_sdf_data, get_local_sdf_data
from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_gazebo.srv import LinkBotStateRequest, LinkBotTrajectoryRequest
from link_bot_planning import shooting_rrt, model_utils
from link_bot_planning.goals import sample_goal
from visual_mpc import gazebo_trajectory_execution

tf.enable_eager_execution()


def collect_classifier_data(args):
    rospy.init_node('collect_classifier_data')

    fwd_model = model_utils.load_generic_model(args.model_dir, args.model_type)
    # TODO: put this inside the generic model loader
    model_path_info = args.model_dir.parts[1:]

    dt = fwd_model.dt

    assert args.env_w >= args.sdf_w
    assert args.env_h >= args.sdf_h

    full_output_directory = random_environment_data_utils.data_directory(args.outdir, *model_path_info)
    full_output_directory = pathlib.Path(full_output_directory)
    if not full_output_directory.is_dir():
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'dt': dt,
            'args': dict([(k, str(v)) for k, v in vars(args).items()])
        }
        json.dump(options, of, indent=1)

    ###########################################
    #             Constraint Model            #
    ###########################################
    constraint_tf_graph = tf.Graph()
    with constraint_tf_graph.as_default():

        class ConstraintCheckerWrapper:

            def __init__(self):
                pass

            @staticmethod
            def get_graph():
                return constraint_tf_graph

            def __call__(self, np_state):
                p_reject = 0.01
                return p_reject

    rrt = shooting_rrt.ShootingRRT(fwd_model=fwd_model,
                                   constraint_checker_wrapper=ConstraintCheckerWrapper(),
                                   dt=dt,
                                   max_v=args.max_v,
                                   n_state=fwd_model.n_state,
                                   planner_timeout=args.planner_timeout,
                                   env_w=args.env_w,
                                   env_h=args.env_h)

    # Preallocate this array once
    examples = np.ndarray([args.n_examples_per_record], dtype=object)
    example_idx = 0
    current_record_traj_idx = 0

    inital_object_dict = {
        'moving_box1': [2.0, 0],
        'moving_box2': [-1.5, 0],
        'moving_box3': [-0.5, 1],
        'moving_box4': [1.5, - 2],
        'moving_box5': [-1.5, - 2.0],
        'moving_box6': [-0.5, 2.0],
    }
    services = gazebo_utils.setup_gazebo_env(args.verbose, args.real_time_rate, inital_object_dict)
    for traj_idx in range(args.n_envs):
        # generate a new environment by rearranging the obstacles
        objects = ['moving_box{}'.format(i) for i in range(1, 7)]
        gazebo_trajectory_execution.move_objects(services, objects, args.env_w, args.env_h, 'velocity', padding=0.5)

        # generate a bunch of plans to random goals
        state_req = LinkBotStateRequest()

        for plan_idx in range(args.n_targets_per_env):
            # generate a random target
            state = services.get_state(state_req)
            head_idx = state.link_names.index("head")
            rope_configuration = gazebo_utils.points_to_config(state.points)
            head_point = state.points[head_idx]
            tail_goal = sample_goal(args.env_w, args.env_h, head_point, env_padding=0.1)

            # Compute SDF Data after all objects have finished moving
            full_sdf_data = get_sdf_data(args.env_h, args.env_w, args.res, services)

            start = np.expand_dims(np.array(rope_configuration), axis=0)
            tail_goal_point = np.array(tail_goal)

            # plan to that target
            if args.verbose >= 1:
                print(Fore.CYAN + "Planning from {} to {}".format(start, tail_goal_point) + Fore.RESET)
            if args.verbose >= 2:
                # tail start x,y and tail goal x,y
                random_environment_data_utils.publish_markers(args,
                                                              tail_goal_point[0], tail_goal_point[1],
                                                              rope_configuration[0], rope_configuration[1],
                                                              marker_size=0.05)

            planned_actions, planned_path, _ = rrt.plan(start, tail_goal_point, full_sdf_data.sdf, args.verbose)

            traj_req = LinkBotTrajectoryRequest()
            traj_req.dt = dt
            if args.verbose >= 4:
                print("Planned actions: {}".format(planned_actions))
                print("Planned path: {}".format(planned_path))

            for action in planned_actions:
                action_msg = LinkBotVelocityAction()
                action_msg.gripper1_velocity.x = action[0]
                action_msg.gripper1_velocity.y = action[1]
                traj_req.gripper1_traj.append(action_msg)

            # execute the plan, collecting the states that actually occurred
            #  TODO: Consider executing just a few steps, so that our start states don't diverge too much
            if args.verbose >= 2:
                print(Fore.CYAN + "Executing Plan.".format(tail_goal_point) + Fore.RESET)

            traj_res = services.execute_trajectory(traj_req)
            # convert ros message into a T x n_state numpy matrix
            actual_rope_configurations = []
            for configuration in traj_res.actual_path:
                np_config = []
                for point in configuration.points:
                    np_config.append(point.x)
                    np_config.append(point.y)
                actual_rope_configurations.append(np_config)
            actual_rope_configurations = np.array(actual_rope_configurations)

            if args.verbose >= 3:
                # FOR THE TAIL
                anim = fwd_model.animate_predict(prediction=planned_path,
                                                 y_rope_configurations=actual_rope_configurations,
                                                 sdf=full_sdf_data.sdf,
                                                 extent=full_sdf_data.extent)
                plt.show()

            # collect the transition pairs (s_t, s_{t+1}, \hat{s}_t, \hat{s}_{t+1})
            states = actual_rope_configurations[:-1]
            next_states = actual_rope_configurations[1:]
            planned_states = planned_path[:-1]
            planned_next_states = planned_path[1:]
            for (state, next_state, action, planned_state, planned_next_state) in zip(states, next_states, planned_actions,
                                                                                      planned_states, planned_next_states):
                actual_head_point = state[4:6]
                planner_head_point = planned_state[4:6]
                # compute the local SDF, which may be different for the state in the planner and the state in the real rollout
                actual_local_sdf_data = get_local_sdf_data(args.sdf_h, args.sdf_w, actual_head_point, full_sdf_data)
                planner_local_sdf_data = get_local_sdf_data(args.sdf_h, args.sdf_w, planner_head_point, full_sdf_data)

                example = ClassifierDataset.make_serialized_example(actual_local_sdf_data.sdf,
                                                                    actual_local_sdf_data.extent,
                                                                    actual_local_sdf_data.origin,
                                                                    planner_local_sdf_data.sdf,
                                                                    planner_local_sdf_data.extent,
                                                                    planner_local_sdf_data.origin,
                                                                    args.sdf_h,  # meters
                                                                    args.sdf_w,  # meters
                                                                    args.res,
                                                                    state,
                                                                    next_state,
                                                                    action,
                                                                    planned_state,
                                                                    planned_next_state)

                if args.verbose >= 4:
                    plt.figure()
                    plt.imshow(planner_local_sdf_data.image > 0, extent=planner_local_sdf_data.extent, zorder=1, alpha=0.5)
                    plt.imshow(actual_local_sdf_data.image > 0, extent=actual_local_sdf_data.extent, zorder=1, alpha=0.5)
                    plt.scatter(actual_head_point[0], actual_head_point[1], zorder=2)
                    plt.scatter(planner_head_point[0], planner_head_point[1], zorder=3)
                    plt.axis("equal")
                    plt.xlabel("x (m)")
                    plt.ylabel("y (m)")
                    plt.show()

                examples[current_record_traj_idx] = example
                current_record_traj_idx += 1
                example_idx += 1

                if current_record_traj_idx == args.n_examples_per_record:
                    # save to a TF record
                    serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

                    end_example_idx = example_idx
                    start_example_idx = end_example_idx - args.n_examples_per_record
                    record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
                    full_filename = full_output_directory / record_filename
                    writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=args.compression_type)
                    writer.write(serialized_dataset)
                    print("saved {}".format(full_filename))

                    current_record_traj_idx = 0


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("model_type", choices=['gp', 'llnn', 'rigid'], default='gp')
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--n-targets-per-env", type=int, default=10)
    parser.add_argument("--n-examples-per-record", type=int, default=512)
    parser.add_argument("--seed", '-s', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=60.0)
    parser.add_argument("--real-time-rate", type=float, default=1.0)
    parser.add_argument('--res', '-r', type=float, default=0.01, help='size of cells in meters')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    # Even though the arena is 5m, we need extra padding so that we can request a 1x1 meter local sdf at the corners
    parser.add_argument('--env-w', type=float, default=6)
    parser.add_argument('--env-h', type=float, default=6)
    parser.add_argument('--sdf-w', type=float, default=1.0)
    parser.add_argument('--sdf-h', type=float, default=1.0)
    parser.add_argument('--max-v', type=float, default=0.15)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
        print("Using seed: ", args.seed)
    np.random.seed(args.seed)

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    # ou.setLogLevel(ou.LOG_DEBUG)
    ou.setLogLevel(ou.LOG_ERROR)

    collect_classifier_data(args)


if __name__ == '__main__':
    main()
