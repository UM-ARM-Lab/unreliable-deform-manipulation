#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import rospy
import tensorflow
from colorama import Fore

import visual_mpc.gazebo_trajectory_execution
from link_bot_data.video_prediction_dataset_utils import bytes_feature, float_feature, int_feature
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import get_sdf_data
from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_pycommon import link_bot_sdf_utils

opts = tensorflow.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
conf = tensorflow.ConfigProto(gpu_options=opts)
tensorflow.enable_eager_execution(config=conf)

from link_bot_data import random_environment_data_utils
from link_bot_gazebo.srv import WorldControlRequest, LinkBotStateRequest

DT = 0.25  # seconds per time step


def generate_traj(args, services, env_idx, global_t_step, gripper1_target_x, gripper1_target_y):
    state_req = LinkBotStateRequest()
    action_msg = LinkBotVelocityAction()

    # Center SDF at the current head position
    state = services.get_state(state_req)
    head_idx = state.link_names.index("head")
    initial_head_point = np.array([state.points[head_idx].x, state.points[head_idx].y])

    # Compute SDF Data
    gradient, sdf, sdf_response = get_sdf_data(services, env_w=args.env_w, env_h=args.env_h, res=args.res)
    resolution = np.array(sdf_response.res)
    origin = np.array(sdf_response.origin)
    sdf_data = link_bot_sdf_utils.SDF(sdf=sdf, gradient=gradient, resolution=resolution, origin=origin)
    sdf_bounds = link_bot_sdf_utils.bounds_from_env_size(args.sdf_w, args.sdf_h, initial_head_point, sdf_data.resolution,
                                                         sdf_data.origin)
    rmin, rmax, cmin, cmax = sdf_bounds
    local_sdf = sdf_data.sdf[rmin:rmax, cmin:cmax]
    local_sdf_gradient = sdf_data.sdf[rmin:rmax, cmin:cmax]
    # indeces of the world point 0,0 in the local sdf
    local_sdf_origin_indeces_in_full_sdf = np.array([rmin, cmin])
    local_sdf_origin = origin - local_sdf_origin_indeces_in_full_sdf

    feature = {
        # These features don't change over time
        'sdf/sdf': float_feature(local_sdf.flatten()),
        'sdf/gradient': float_feature(local_sdf_gradient.flatten()),
        'sdf/shape': int_feature(np.array(local_sdf.shape)),
        'sdf/resolution': float_feature(sdf_data.resolution),
        'sdf/origin': float_feature(local_sdf_origin),
        'sdf/head_points': float_feature(initial_head_point),
    }

    combined_constraint_labels = np.ndarray((args.steps_per_traj, 1))
    for t in range(args.steps_per_traj):
        # Query the current state
        state = services.get_state(state_req)
        head_idx = state.link_names.index("head")
        rope_configuration = gazebo_utils.points_to_config(state.points)
        head_point = state.points[head_idx]

        # Pick a new target if necessary
        if global_t_step % args.steps_per_target == 0:
            gripper1_target_x, gripper1_target_y = sample_goal(args.env_w, args.env_h, head_point, env_padding=0.25)
            if args.verbose:
                print('gripper target:', gripper1_target_x, gripper1_target_y)
                random_environment_data_utils.publish_marker(args, gripper1_target_x, gripper1_target_y, marker_size=0.05)

        image = np.copy(np.frombuffer(state.camera_image.data, dtype=np.uint8)).reshape([64, 64, 3])

        # compute the velocity to move in that direction
        velocity = np.minimum(np.maximum(np.random.randn() * 0.07 + 0.10, 0), 0.15)
        gripper1_target_vx = velocity if gripper1_target_x > head_point.x else -velocity
        gripper1_target_vy = velocity if gripper1_target_y > head_point.y else -velocity

        # publish the pull command, which will return the target velocity
        action_msg.gripper1_velocity.x = gripper1_target_vx
        action_msg.gripper1_velocity.y = gripper1_target_vy
        services.velocity_action_pub.publish(action_msg)

        # let the simulator run
        step = WorldControlRequest()
        step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
        services.world_control(step)  # this will block until stepping is complete

        post_action_state = services.get_state(state_req)
        stopped = 0.025  # This threshold must be tuned whenever physics or the above velocity controller parameters change
        if (abs(post_action_state.gripper1_velocity.x) < stopped < abs(gripper1_target_vx)) \
                or (abs(post_action_state.gripper1_velocity.y) < stopped < abs(gripper1_target_vy)):
            at_constraint_boundary = True
        else:
            at_constraint_boundary = False

        if args.verbose:
            print("{} {:0.4f} {:0.4f} {:0.4f} {:0.4f} {}".format(t,
                                                                 gripper1_target_vx,
                                                                 gripper1_target_vy,
                                                                 post_action_state.gripper1_velocity.x,
                                                                 post_action_state.gripper1_velocity.y,
                                                                 at_constraint_boundary))

        combined_constraint_labels[t, 0] = at_constraint_boundary

        # format the tf feature
        feature['{}/image_aux1/encoded'.format(t)] = bytes_feature(image.tobytes())
        feature['{}/endeffector_pos'.format(t)] = float_feature(np.array([head_point.x, head_point.y]))
        feature['{}/1/velocity'.format(t)] = float_feature(np.array([state.gripper1_velocity.x, state.gripper1_velocity.y]))
        feature['{}/1/post_action_velocity'.format(t)] = float_feature(np.array([post_action_state.gripper1_velocity.x,
                                                                                 post_action_state.gripper1_velocity.y]))
        feature['{}/1/force'.format(t)] = float_feature(np.array([state.gripper1_force.x, state.gripper1_force.y]))
        feature['{}/rope_configuration'.format(t)] = float_feature(rope_configuration.flatten())
        feature['{}/action'.format(t)] = float_feature(np.array([gripper1_target_vx, gripper1_target_vy]))
        feature['{}/constraint'.format(t)] = float_feature(np.array([float(at_constraint_boundary)]))

        global_t_step += 1

    n_positive = np.count_nonzero(np.any(combined_constraint_labels, axis=1))
    percentage_positive = n_positive * 100.0 / combined_constraint_labels.shape[0]

    if args.verbose:
        print(Fore.GREEN + "Trajectory {} Complete".format(env_idx) + Fore.RESET)

    example_proto = tensorflow.train.Example(features=tensorflow.train.Features(feature=feature))
    # TODO: include documentation *inside* the tfrecords file describing what each feature is
    example = example_proto.SerializeToString()
    return example, percentage_positive, global_t_step, gripper1_target_x, gripper1_target_y


def sample_goal(w, h, current_head_point, env_padding):
    gripper1_current_x = current_head_point.x
    gripper1_current_y = current_head_point.y
    current = np.array([gripper1_current_x, gripper1_current_y])
    min_near = 0.5
    while True:
        gripper1_target_x = np.random.uniform(-w / 2 + env_padding, w / 2 - env_padding)
        gripper1_target_y = np.random.uniform(-h / 2 + env_padding, h / 2 - env_padding)
        target = np.array([gripper1_target_x, gripper1_target_y])
        d = np.linalg.norm(current - target)
        if d > min_near:
            break
    return gripper1_target_x, gripper1_target_y


def generate_trajs(args, full_output_directory, services):
    examples = np.ndarray([args.n_trajs_per_file], dtype=object)
    percentages_positive = []
    global_t_step = 0
    gripper1_target_x = None
    gripper1_target_y = None
    for i in range(args.n_trajs):
        current_record_traj_idx = i % args.n_trajs_per_file

        if not args.no_obstacles:
            visual_mpc.gazebo_trajectory_execution.move_objects(services, args.env_w, args.env_h, 'velocity', padding=0.1)

        # Generate a new trajectory
        example, percentage_violation, global_t_step, gripper1_target_x, gripper1_target_y = generate_traj(args, services, i,
                                                                                                           global_t_step,
                                                                                                           gripper1_target_x,
                                                                                                           gripper1_target_y)
        examples[current_record_traj_idx] = example
        percentages_positive.append(percentage_violation)

        # Save the data
        if current_record_traj_idx == args.n_trajs_per_file - 1:
            # Construct the dataset where each trajectory has been serialized into one big string
            # since tfrecords don't really support hierarchical data structures
            serialized_dataset = tensorflow.data.Dataset.from_tensor_slices((examples))

            end_traj_idx = i + args.start_idx_offset
            start_traj_idx = end_traj_idx - args.n_trajs_per_file + 1
            full_filename = os.path.join(full_output_directory, "traj_{}_to_{}.tfrecords".format(start_traj_idx, end_traj_idx))
            writer = tensorflow.data.experimental.TFRecordWriter(full_filename, compression_type=args.compression_type)
            writer.write(serialized_dataset)
            print("saved {}".format(full_filename))

            if args.verbose:
                mean_percentage_positive = np.mean(percentages_positive)
                print("Class balance: mean % positive: {}".format(mean_percentage_positive))

        if not args.verbose:
            print(".", end='')
            sys.stdout.flush()


def generate(args):
    rospy.init_node('gazebo_small_with_images')

    assert args.n_trajs % args.n_trajs_per_file == 0, "num trajs must be multiple of {}".format(args.n_trajs_per_file)
    assert args.env_w >= args.sdf_w
    assert args.env_h >= args.sdf_h

    full_output_directory = random_environment_data_utils.data_directory(args.outdir, args.n_trajs)
    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'dt': DT,
            'sdf_w': args.sdf_w,
            'sdf_h': args.sdf_h,
            'env_w': args.env_w,
            'env_h': args.env_h,
            'compression_type': args.compression_type
        }
        json.dump(options, of, indent=1)

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
        print("Using seed: ", args.seed)
    np.random.seed(args.seed)

    services = gazebo_utils.setup_gazebo_env(args.verbose, args.real_time_rate)

    generate_trajs(args, full_output_directory, services)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=5000)
    tensorflow.logging.set_verbosity(tensorflow.logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("n_trajs", help='how many trajectories to collect', type=int)
    parser.add_argument("outdir")
    parser.add_argument('--res', '-r', type=float, default=0.01, help='size of cells in meters')
    parser.add_argument('--env-w', type=float, default=1.0)
    parser.add_argument('--env-h', type=float, default=1.0)
    parser.add_argument('--sdf-w', type=float, default=1.0)
    parser.add_argument('--sdf-h', type=float, default=1.0)
    parser.add_argument("--steps-per-traj", type=int, default=100)
    parser.add_argument("--steps-per-target", type=int, default=25)
    parser.add_argument("--start-idx-offset", type=int, default=0)
    parser.add_argument("--no-obstacles", action='store_true')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument("--n-trajs-per-file", type=int, default=512)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--real-time-rate", help='number of times real time', type=float, default=10)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
