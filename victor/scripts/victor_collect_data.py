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
from link_bot_gazebo.srv import LinkBotStateRequest, ExecuteActionRequest

from link_bot_data import random_environment_data_utils
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature
from victor import victor_utils
from link_bot_planning.params import LocalEnvParams, FullEnvParams, SimParams
from link_bot_pycommon import ros_pycommon, link_bot_pycommon
from link_bot_pycommon.args import my_formatter

opts = tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
conf = tensorflow.compat.v1.ConfigProto(gpu_options=opts)
tensorflow.compat.v1.enable_eager_execution(config=conf)


def sample_delta_pos(action_rng, max_delta_pos, head_point, goal_env_w, goal_env_h):
    """

    :param action_rng:
    :param max_delta_pos:
    :param head_point:
    :param goal_env_w: this is HALF the env width, assumed to be centered at 0
    :param goal_env_h: this is HALF the env height, assumed to be centered at 0
    :return:
    """
    while True:
        delta_pos = action_rng.uniform(0, max_delta_pos)
        direction = action_rng.uniform(-np.pi, np.pi)
        gripper1_dx = np.cos(direction) * delta_pos
        gripper1_dy = np.sin(direction) * delta_pos

        if -goal_env_w <= head_point.x + gripper1_dx <= goal_env_w and -goal_env_h <= head_point.y + gripper1_dy <= goal_env_h:
            break

    return gripper1_dx, gripper1_dy

def generate_traj(args, services, traj_idx, global_t_step, action_rng: np.random.RandomState):
    state_req = LinkBotStateRequest()
    action_msg = ExecuteActionRequest()

    max_delta_pos = ros_pycommon.get_max_speed() * args.dt

    # At this point, we hope all of the objects have stopped moving, so we can get the environment and assume it never changes
    # over the course of this function
    full_env_data = ros_pycommon.get_occupancy_data(env_w=args.env_w, env_h=args.env_h, res=args.res, services=services)

    feature = {
        'local_env_rows': float_tensor_to_bytes_feature([args.local_env_rows]),
        'local_env_cols': float_tensor_to_bytes_feature([args.local_env_cols]),
        'full_env/env': float_tensor_to_bytes_feature(full_env_data.data),
        'full_env/extent': float_tensor_to_bytes_feature(full_env_data.extent),
        'full_env/origin': float_tensor_to_bytes_feature(full_env_data.origin),
    }

    for time_idx in range(args.steps_per_traj):
        # Query the current state
        state = services.get_state(state_req)
        head_idx = state.link_names.index("head")
        points_flat = link_bot_pycommon.flatten_points(state.points)
        head_point = state.points[head_idx]

        gripper1_dx, gripper1_dy = sample_delta_pos(action_rng, max_delta_pos, head_point, args.goal_env_w, args.goal_env_h)
        if args.verbose >= 2:
            print('gripper delta:', gripper1_dx, gripper1_dy)
            random_environment_data_utils.publish_marker(head_point.x + gripper1_dx, head_point.y + gripper1_dy, marker_size=0.05)

        action_msg.action.gripper1_delta_pos.x = gripper1_dx
        action_msg.action.gripper1_delta_pos.y = gripper1_dy
        action_msg.action.max_time_per_step = args.dt
        services.execute_action(action_msg)

        # format the tf feature
        head_np = np.array([head_point.x, head_point.y])
        local_env_data = ros_pycommon.get_local_occupancy_data(args.local_env_rows,
                                                               args.local_env_cols,
                                                               args.res,
                                                               center_point=head_np,
                                                               services=services)

        feature['{}/action'.format(time_idx)] = float_tensor_to_bytes_feature([gripper1_dx, gripper1_dy])
        feature['{}/state/link_bot'.format(time_idx)] = float_tensor_to_bytes_feature(points_flat)
        feature['{}/state/local_env'.format(time_idx)] = float_tensor_to_bytes_feature(local_env_data.data)
        feature['{}/state/local_env_origin'.format(time_idx)] = float_tensor_to_bytes_feature(local_env_data.origin)
        feature['{}/res'.format(time_idx)] = float_tensor_to_bytes_feature(local_env_data.resolution[0])
        feature['{}/traj_idx'.format(time_idx)] = float_tensor_to_bytes_feature(traj_idx)
        feature['{}/time_idx'.format(time_idx)] = float_tensor_to_bytes_feature(time_idx)

        global_t_step += 1

    if args.verbose >= 2:
        print(Fore.GREEN + "Trajectory {} Complete".format(traj_idx) + Fore.RESET)

    example_proto = tensorflow.train.Example(features=tensorflow.train.Features(feature=feature))
    example = example_proto.SerializeToString()
    return example, global_t_step


def generate_trajs(args, full_output_directory, services, env_rng: np.random.RandomState, action_rng: np.random.RandomState):
    examples = np.ndarray([args.trajs_per_file], dtype=object)
    global_t_step = 0
    for i in range(args.trajs):
        current_record_traj_idx = i % args.trajs_per_file

        # Generate a new trajectory
        example, global_t_step = generate_traj(args, services, i, global_t_step, action_rng)
        examples[current_record_traj_idx] = example

        # Save the data
        if current_record_traj_idx == args.trajs_per_file - 1:
            # Construct the dataset where each trajectory has been serialized into one big string
            # since tfrecords don't really support hierarchical data structures
            serialized_dataset = tensorflow.data.Dataset.from_tensor_slices((examples))

            end_traj_idx = i + args.start_idx_offset
            start_traj_idx = end_traj_idx - args.trajs_per_file + 1
            full_filename = os.path.join(full_output_directory, "traj_{}_to_{}.tfrecords".format(start_traj_idx, end_traj_idx))
            writer = tensorflow.data.experimental.TFRecordWriter(full_filename, compression_type=args.compression_type)
            writer.write(serialized_dataset)
            print("saved {}".format(full_filename))

        if args.verbose == 1:
            print(".", end='')
            sys.stdout.flush()


def generate(args):
    rospy.init_node('victor_collect_dynamics_data')

    n_state = ros_pycommon.get_n_state()
    rope_length = ros_pycommon.get_rope_length()

    assert args.trajs % args.trajs_per_file == 0, "num trajs must be multiple of {}".format(args.trajs_per_file)

    full_output_directory = random_environment_data_utils.data_directory(args.outdir, args.trajs)
    if not os.path.isdir(full_output_directory) and args.verbose >= 1:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    local_env_params = LocalEnvParams(h_rows=args.local_env_rows, w_cols=args.local_env_cols, res=args.res)
    full_env_cols = int(args.env_w / args.res)
    full_env_rows = int(args.env_h / args.res)
    full_env_params = FullEnvParams(h_rows=full_env_rows, w_cols=full_env_cols, res=args.res)
    sim_params = SimParams(real_time_rate=args.real_time_rate,
                           max_step_size=args.max_step_size,
                           goal_padding=0.5,
                           move_obstacles=(not args.no_obstacles),
                           nudge=False)
    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'dt': args.dt,
            'max_step_size': args.max_step_size,
            'rope_length': rope_length,
            'local_env_params': local_env_params.to_json(),
            'full_env_params': full_env_params.to_json(),
            'sim_params': sim_params.to_json(),
            'compression_type': args.compression_type,
            'sequence_length': args.steps_per_traj,
            'n_state': n_state,
            'n_action': 2,
        }
        json.dump(options, of, indent=1)

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    print(Fore.CYAN + "Using seed: {}".format(args.seed) + Fore.RESET)
    np.random.seed(args.seed)
    env_rng = np.random.RandomState(args.seed)
    goal_rng = np.random.RandomState(args.seed)

    services = victor_utils.setup_env(args.verbose, reset_world=True)
    # services = victor_utils.setup_env(args.verbose, args.real_time_rate, args.max_step_size, True, None)

    generate_trajs(args, full_output_directory, services, env_rng, goal_rng)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=5000)
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.DEBUG)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("trajs", type=int, help='how many trajectories to collect')
    parser.add_argument("outdir")
    parser.add_argument('--dt', type=float, default=1.00, help='seconds to execute each delta position action')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument('--env-w', type=float, default=6.0, help='full env w')
    parser.add_argument('--env-h', type=float, default=6.0, help='full env h')
    parser.add_argument('--goal-env-w', type=float, default=2.2, help='goal env w')
    parser.add_argument('--goal-env-h', type=float, default=2.2, help='goal env h')
    parser.add_argument('--local_env-cols', type=int, default=50, help='local env')
    parser.add_argument('--local_env-rows', type=int, default=50, help='local env')
    parser.add_argument("--steps-per-traj", type=int, default=100, help='steps per traj')
    parser.add_argument("--start-idx-offset", type=int, default=0, help='offset TFRecord file names')
    parser.add_argument("--move-objects-every-n", type=int, default=16, help='rearrange objects every n trajectories')
    parser.add_argument("--no-obstacles", action='store_true', help='do not move obstacles')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB', help='compression type')
    parser.add_argument("--trajs-per-file", type=int, default=128, help='trajs per file')
    parser.add_argument("--seed", '-s', type=int, help='seed')
    parser.add_argument("--real-time-rate", type=float, default=0, help='number of times real time')
    parser.add_argument("--max-step-size", type=float, default=0.01, help='seconds per physics step')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
