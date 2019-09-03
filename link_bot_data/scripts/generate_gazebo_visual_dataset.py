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
from link_bot_pycommon.link_bot_sdf_utils import SDF

opts = tensorflow.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
conf = tensorflow.ConfigProto(gpu_options=opts)
tensorflow.enable_eager_execution(config=conf)

from link_bot_data import random_environment_data_utils
from link_bot_gazebo.msg import MultiLinkBotPositionAction, Position2dEnable
from link_bot_gazebo.srv import WorldControlRequest, LinkBotStateRequest

DT = 0.25  # seconds per time step
env_w = 1
env_h = 1


def generate_traj(args, services, env_idx):
    state_req = LinkBotStateRequest()
    action_msg = MultiLinkBotPositionAction()

    # Compute SDF Data
    gradient, sdf, sdf_response = get_sdf_data(services)
    res = np.array(sdf_response.res)
    origin = np.array(sdf_response.origin)
    sdf_data = SDF(sdf=sdf, gradient=gradient, resolution=res, origin=origin)

    # bias sampling to explore by choosing a target location at least min_near away
    gripper1_target_x, gripper1_target_y = sample_goal(services, state_req)

    if args.verbose:
        print('gripper target:', gripper1_target_x, gripper1_target_y)
        random_environment_data_utils.publish_marker(args, gripper1_target_x, gripper1_target_y)

    feature = {
        # These features don't change over time
        'sdf/sdf': float_feature(sdf_data.sdf.flatten()),
        'sdf/gradient': float_feature(sdf_data.gradient.flatten()),
        'sdf/shape': int_feature(np.array(sdf_data.sdf.shape)),
        'sdf/resolution': float_feature(sdf_data.resolution),
        'sdf/origin': float_feature(sdf_data.origin.astype(np.float32))
    }
    combined_constraint_labels = np.ndarray((args.steps_per_traj, 1))
    last_rope_configuration = None
    for t in range(args.steps_per_traj):
        if t % args.steps_per_target == 0:
            gripper1_target_x, gripper1_target_y = sample_goal(services, state_req)
            if args.verbose:
                print('gripper target:', gripper1_target_x, gripper1_target_y)
                random_environment_data_utils.publish_marker(args, gripper1_target_x, gripper1_target_y)

        # publish the pull command
        action_msg.gripper1_pos.x = gripper1_target_x
        action_msg.gripper1_pos.y = gripper1_target_y
        services.position_action_pub.publish(action_msg)

        s = services.get_state(state_req)
        image = np.copy(np.frombuffer(s.camera_image.data, dtype=np.uint8)).reshape([64, 64, 3])

        # Note: ground truth labels are just based on force/velocity
        target_velocity = [s.gripper1_target_velocity.x,
                           s.gripper1_target_velocity.y,
                           s.gripper1_target_velocity.z]
        current_velocity = [s.gripper1_velocity.x,
                            s.gripper1_velocity.y,
                            s.gripper1_velocity.z]
        target_speed = np.linalg.norm(target_velocity)
        speed = np.linalg.norm(current_velocity)
        if abs(target_speed - speed) > 0.05:
            at_constraint_boundary = True
        else:
            at_constraint_boundary = False

        if args.verbose:
            print(t, abs(speed - target_speed), at_constraint_boundary)

        combined_constraint_labels[t, 0] = at_constraint_boundary

        # format the tf feature
        rope_configuration = np.array([[pt.x, pt.y] for pt in s.points])
        head_idx = s.link_names.index("head")
        feature['{}/image_aux1/encoded'.format(t)] = bytes_feature(image.tobytes())
        feature['{}/endeffector_pos'.format(t)] = float_feature(np.array([s.points[head_idx].x, s.points[head_idx].y]))
        feature['{}/1/velocity'.format(t)] = float_feature(np.array([s.gripper1_velocity.x, s.gripper1_velocity.y]))
        feature['{}/1/target_velocity'.format(t)] = float_feature(
            np.array([s.gripper1_target_velocity.x, s.gripper1_target_velocity.y]))
        feature['{}/1/force'.format(t)] = float_feature(np.array([s.gripper1_force.x, s.gripper1_force.y]))
        feature['{}/constraint'.format(t)] = float_feature(np.array([float(at_constraint_boundary)]))
        feature['{}/rope_configuration'.format(t)] = float_feature(rope_configuration.flatten())

        # let the simulator run
        step = WorldControlRequest()
        # TODO: save this information INTO THE DATASET HPARAMS!!!
        step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
        services.world_control(step)  # this will block until stepping is complete

        # Get the action now that it has been computed in the gazebo Update
        s = services.get_state(state_req)
        feature['{}/action'.format(t)] = float_feature(np.array([s.gripper1_target_velocity.x, s.gripper1_target_velocity.y]))

        # plt.imshow(image, extent=[-0.55, 0.55, -0.55, 0.55])
        # plt.imshow(sdf_data.image > 0, extent=[-0.55, 0.55, -0.55, 0.55])
        # plt.title(np.array2string(rope_configuration))
        # if last_rope_configuration is not None:
        #     plt.plot([last_rope_configuration[0, 0], last_rope_configuration[1, 0], last_rope_configuration[2, 0]],
        #              [last_rope_configuration[0, 1], last_rope_configuration[1, 1], last_rope_configuration[2, 1]], c='r')
        # plt.plot([rope_configuration[0, 0], rope_configuration[1, 0], rope_configuration[2, 0]],
        #          [rope_configuration[0, 1], rope_configuration[1, 1], rope_configuration[2, 1]], c='g')
        # plt.quiver(rope_configuration[2, 0], rope_configuration[2, 1], s.gripper1_target_velocity.x, s.gripper1_target_velocity.y)
        # plt.show()

        last_rope_configuration = rope_configuration

    n_positive = np.count_nonzero(np.any(combined_constraint_labels, axis=1))
    percentage_positive = n_positive * 100.0 / combined_constraint_labels.shape[0]

    if args.verbose:
        print(Fore.GREEN + "Trajectory {} Complete".format(env_idx) + Fore.RESET)

    example_proto = tensorflow.train.Example(features=tensorflow.train.Features(feature=feature))
    # TODO: include documentation *inside* the tfrecords file describing what each feature is
    example = example_proto.SerializeToString()
    return example, percentage_positive


def sample_goal(services, state_req):
    s = services.get_state(state_req)
    head_idx = s.link_names.index('head')
    current_head_point = s.points[head_idx]
    gripper1_current_x = current_head_point.x
    gripper1_current_y = current_head_point.y
    current = np.array([gripper1_current_x, gripper1_current_y])
    min_near = 0.5
    while True:
        gripper1_target_x = np.random.uniform(-env_w / 2, env_w / 2)
        gripper1_target_y = np.random.uniform(-env_h / 2, env_h / 2)
        target = np.array([gripper1_target_x, gripper1_target_y])
        d = np.linalg.norm(current - target)
        if d > min_near:
            break
    return gripper1_target_x, gripper1_target_y


def generate_trajs(args, full_output_directory, services):
    examples = np.ndarray([args.n_trajs_per_file], dtype=object)
    percentages_positive = []
    for i in range(args.n_trajs):
        current_record_traj_idx = i % args.n_trajs_per_file

        visual_mpc.gazebo_trajectory_execution.move_objects(services, env_w, env_h, 'position')

        # Generate a new trajectory
        example, percentage_violation = generate_traj(args, services, i)
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

    full_output_directory = random_environment_data_utils.data_directory(args.outdir, args.n_trajs)
    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    with open(pathlib.Path(full_output_directory) / 'option.json', 'w') as of:
        options = {
            'dt': DT,
            'env_w': env_w,
            'env_h': env_h,
        }
        json.dump(options, of)

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
    parser.add_argument("--steps-per-traj", type=int, default=100)
    parser.add_argument("--steps-per-target", type=int, default=16)
    parser.add_argument("--start-idx-offset", type=int, default=0)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument("--n-trajs-per-file", type=int, default=512)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--real-time-rate", help='number of times real time', type=float, default=10)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
