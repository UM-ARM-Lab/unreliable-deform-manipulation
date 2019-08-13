#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import os
import sys

import numpy as np
import rospy
import tensorflow
from colorama import Fore
from std_msgs.msg import String
from std_srvs.srv import EmptyRequest
from tf.transformations import quaternion_from_euler

from link_bot_data.video_prediction_dataset_utils import bytes_feature, float_feature, int_feature
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_sdf_tools.srv import ComputeSDF

opts = tensorflow.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
conf = tensorflow.ConfigProto(gpu_options=opts)
tensorflow.enable_eager_execution(config=conf)

from gazebo_msgs.srv import GetPhysicsPropertiesRequest
from gazebo_msgs.srv import SetPhysicsPropertiesRequest
from link_bot_data import random_environment_data_utils
from link_bot_gazebo.msg import MultiLinkBotPositionAction, Position2dEnable, Position2dAction, ObjectAction
from link_bot_gazebo.srv import WorldControlRequest, LinkBotStateRequest
from link_bot_sdf_tools import link_bot_sdf_tools

DT = 0.25  # seconds per time step
w = 1
h = 1


def generate_traj(args, services, env_idx):
    state_req = LinkBotStateRequest()
    action_msg = MultiLinkBotPositionAction()

    # Compute SDF Data
    sdf_data = link_bot_sdf_tools.request_sdf_data(services.compute_sdf, width=w, height=h, res=args.res)

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
    for t in range(args.steps_per_traj):
        if t % args.steps_per_target == 0:
            gripper1_target_x, gripper1_target_y = sample_goal(services, state_req)
            if args.verbose:
                print('gripper target:', gripper1_target_x, gripper1_target_y)
                random_environment_data_utils.publish_marker(args, gripper1_target_x, gripper1_target_y)

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

        # publish the pull command
        action_msg.gripper1_pos.x = gripper1_target_x
        action_msg.gripper1_pos.y = gripper1_target_y
        services.position_action_pub.publish(action_msg)

        # format the tf feature
        head_idx = s.link_names.index("head")
        feature['{}/image_aux1/encoded'.format(t)] = bytes_feature(image.tobytes())
        feature['{}/endeffector_pos'.format(t)] = float_feature(np.array([s.points[head_idx].x, s.points[head_idx].y]))
        feature['{}/action'.format(t)] = float_feature(np.array([s.gripper1_target_velocity.x, s.gripper1_target_velocity.y]))
        feature['{}/1/velocity'.format(t)] = float_feature(np.array([s.gripper1_velocity.x, s.gripper1_velocity.y]))
        feature['{}/1/force'.format(t)] = float_feature(np.array([s.gripper1_force.x, s.gripper1_force.y]))
        feature['{}/constraint'.format(t)] = float_feature(np.array([float(at_constraint_boundary)]))
        feature['{}/rope_configuration'.format(t)] = float_feature(np.array([[pt.x, pt.y] for pt in s.points]).flatten())

        # let the simulator run
        step = WorldControlRequest()
        step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
        services.world_control(step)  # this will block until stepping is complete

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
        gripper1_target_x = np.random.uniform(-w / 2, w / 2)
        gripper1_target_y = np.random.uniform(-h / 2, h / 2)
        target = np.array([gripper1_target_x, gripper1_target_y])
        d = np.linalg.norm(current - target)
        if d > min_near:
            break
    return gripper1_target_x, gripper1_target_y


def random_object_move(model_name):
    move = ObjectAction()
    move.pose.position.x = np.random.uniform(-w / 2 + 0.05, w / 2 - 0.05)
    move.pose.position.y = np.random.uniform(-w / 2 + 0.05, w / 2 - 0.05)
    q = quaternion_from_euler(0, 0, np.random.uniform(-np.pi, np.pi))
    move.pose.orientation.x = q[0]
    move.pose.orientation.y = q[1]
    move.pose.orientation.z = q[2]
    move.pose.orientation.w = q[3]
    move.model_name = model_name
    return move


def generate_trajs(args, full_output_directory, services):
    # construct message we will publish
    enable_objects = Position2dEnable()
    enable_objects.model_names = ['cheezits_box', 'tissue_box']
    enable_objects.enable = True

    disable_objects = Position2dEnable()
    disable_objects.model_names = ['cheezits_box', 'tissue_box']
    disable_objects.enable = False

    disable_link_bot = String()
    disable_link_bot.data = 'disabled'

    enable_link_bot = String()
    enable_link_bot.data = 'position'

    examples = np.ndarray([args.n_trajs_per_file], dtype=object)
    percentages_positive = []
    for i in range(args.n_trajs):
        current_record_traj_idx = i % args.n_trajs_per_file

        # disable the rope controller, enable the hand-of-god to move objects
        services.position_2d_enable.publish(enable_objects)
        services.link_bot_mode.publish(disable_link_bot)

        # Move the objects
        move_action = Position2dAction()
        cheezits_move = random_object_move("cheezits_box")
        tissue_move = random_object_move("tissue_box")
        move_action.actions.append(cheezits_move)
        move_action.actions.append(tissue_move)
        services.position_2d_action.publish(move_action)

        # let the move actually occur
        step = WorldControlRequest()
        move_wait_duration = 5
        step.steps = int(move_wait_duration / 0.001)  # assuming 0.001s per simulation step
        services.world_control(step)  # this will block until stepping is complete

        # disable the objects so they stop, enabled the rope controller
        services.position_2d_enable.publish(disable_objects)
        services.link_bot_mode.publish(enable_link_bot)

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

    if not args.seed:
        args.seed = np.random.randint(0, 10000)
        print("Using seed: ", args.seed)
    np.random.seed(args.seed)

    # fire up services
    services = GazeboServices()
    services.compute_sdf = rospy.ServiceProxy('/sdf', ComputeSDF)
    services.wait(args)
    empty = EmptyRequest()
    services.reset.call(empty)

    # set up physics
    get = GetPhysicsPropertiesRequest()
    current_physics = services.get_physics.call(get)
    set = SetPhysicsPropertiesRequest()
    set.gravity = current_physics.gravity
    set.time_step = current_physics.time_step
    set.ode_config = current_physics.ode_config
    set.max_update_rate = args.real_time_rate * 1000.0
    set.enabled = True
    services.set_physics.call(set)

    # Set initial object positions
    move_action = Position2dAction()
    cheezits_move = ObjectAction()
    cheezits_move.pose.position.x = 0.20
    cheezits_move.pose.position.y = -0.25
    cheezits_move.pose.orientation.x = 0
    cheezits_move.pose.orientation.y = 0
    cheezits_move.pose.orientation.z = 0
    cheezits_move.pose.orientation.w = 0
    cheezits_move.model_name = "cheezits_box"
    move_action.actions.append(cheezits_move)
    tissue_move = ObjectAction()
    tissue_move.pose.position.x = 0.20
    tissue_move.pose.position.y = 0.25
    tissue_move.pose.orientation.x = 0
    tissue_move.pose.orientation.y = 0
    tissue_move.pose.orientation.z = 0
    tissue_move.pose.orientation.w = 0
    tissue_move.model_name = "tissue_box"
    move_action.actions.append(tissue_move)
    services.position_2d_action.publish(move_action)

    # let the simulator run to get the first image
    step = WorldControlRequest()
    step.steps = int(5.0 / 0.001)  # assuming 0.001s per simulation step
    services.world_control(step)  # this will block until stepping is complete

    generate_trajs(args, full_output_directory, services)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=5000)
    tensorflow.logging.set_verbosity(tensorflow.logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("n_trajs", help='how many trajectories to collect', type=int)
    parser.add_argument("outdir")
    parser.add_argument('--res', '-r', type=float, default=0.01, help='size of cells in meters')
    parser.add_argument("--steps-per-traj", type=int, default=100)
    parser.add_argument("--steps-per-target", type=int, default=7)
    parser.add_argument("--start-idx-offset", type=int, default=0)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument("--n-trajs-per-file", type=int, default=128)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--real-time-rate", help='number of times real time', type=float, default=10)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
