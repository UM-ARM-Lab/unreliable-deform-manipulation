#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import os
import tensorflow as tf
import sys

import numpy as np
import rospy
from link_bot_data import video_prediction_dataset_utils
from colorama import Fore

from std_srvs.srv import Empty, EmptyRequest
from link_bot_data import random_environment_data_utils
from link_bot_data.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_gazebo.msg import LinkBotConfiguration, MultiLinkBotPositionAction
from link_bot_sdf_tools.srv import ComputeSDF
from gazebo_msgs.srv import GetPhysicsProperties, GetPhysicsPropertiesRequest, GetPhysicsPropertiesResponse
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest, SetPhysicsPropertiesResponse
from link_bot_gazebo.srv import WorldControl, WorldControlRequest, LinkBotState, LinkBotStateRequest
from link_bot_models.label_types import LabelType
from link_bot_sdf_tools import link_bot_sdf_tools

DT = 0.1  # seconds per time step


class GazeboServices:

    def __init__(self):
        self.action_pub = rospy.Publisher("/multi_link_bot_position_action", MultiLinkBotPositionAction, queue_size=10)
        self.config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
        self.world_control = rospy.ServiceProxy('/world_control', WorldControl)
        self.get_state = rospy.ServiceProxy('/link_bot_state', LinkBotState)
        self.compute_sdf = rospy.ServiceProxy('/sdf', ComputeSDF)
        self.get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        self.reset = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.services_to_wait_for = [
            '/world_control',
            '/link_bot_state',
            '/sdf',
            '/gazebo/get_physics_properties',
            '/gazebo/set_physics_properties',
            '/gazebo/reset_simulation',
        ]

    def wait(self, args):
        if args.verbose:
            print(Fore.CYAN + "Waiting for services..." + Fore.RESET)
        for s in self.services_to_wait_for:
            rospy.wait_for_service(s)
        if args.verbose:
            print(Fore.CYAN + "Done waiting for services" + Fore.RESET)



def generate_tf_record():
    pass


def generate_traj(args, services, env_idx):
    state_req = LinkBotStateRequest()
    action_msg = MultiLinkBotPositionAction()

    # Compute SDF Data
    w = 1
    h = 1
    sdf_data = link_bot_sdf_tools.request_sdf_data(services.compute_sdf, width=w, height=h, res=args.res)

    # Create random rope configurations by picking a random point and applying forces to move the rope to that point
    rope_configurations = np.ndarray((args.steps_per_traj, 6), dtype=np.float32)
    images = np.ndarray(args.steps_per_traj, dtype=object)
    gripper1_forces = np.ndarray((args.steps_per_traj, 2))
    gripper1_target_velocities = np.ndarray((args.steps_per_traj, 2))
    gripper1_velocities = np.ndarray((args.steps_per_traj, 2))
    gripper2_forces = np.ndarray((args.steps_per_traj, 2))
    gripper2_target_velocities = np.ndarray((args.steps_per_traj, 2))
    gripper2_velocities = np.ndarray((args.steps_per_traj, 2))
    combined_constraint_labels = np.ndarray((args.steps_per_traj, 1), dtype=np.float32)

    gripper1_target_x = np.random.uniform(-w / 2, w / 2)
    gripper1_target_y = np.random.uniform(-h / 2, h / 2)
    gripper2_target_x = 0
    gripper2_target_y = 0

    if args.verbose:
        random_environment_data_utils.publish_marker(args, gripper1_target_x, gripper1_target_y)

    for t in range(args.steps_per_traj):
        # save the state and action data
        s = services.get_state(state_req)
        rope_configurations[t] = np.array([[pt.x, pt.y] for pt in s.points]).flatten()

        # FIXME: this is a hack
        while True:
            s = services.get_state(state_req)
            step = WorldControlRequest()
            step.steps = int(0.05 / 0.001)  # assuming 0.001s per simulation step
            services.world_control(step)  # this will block until stepping is complete
            image = np.frombuffer(s.camera_image.data, dtype=np.uint8)
            if image.size == 64*64*3:
                break

        images[t] = image.reshape(64, 64, 3).tobytes()
        gripper1_forces[t] = np.array([s.gripper1_force.x, s.gripper1_force.y])
        gripper1_target_velocities[t] = np.array([s.gripper1_target_velocity.x, s.gripper1_target_velocity.y])
        gripper1_velocities[t] = np.array([s.gripper1_velocity.x, s.gripper1_velocity.y])
        gripper2_forces[t] = np.array([s.gripper1_force.x, s.gripper1_force.y])
        gripper2_target_velocities[t] = np.array([s.gripper1_target_velocity.x, s.gripper1_target_velocity.y])
        gripper2_velocities[t] = np.array([s.gripper1_velocity.x, s.gripper1_velocity.y])

        # TODO: use ground truth labels not just based on force/velocity?
        target_velocity = [s.gripper1_target_velocity.x,
                           s.gripper1_target_velocity.y,
                           s.gripper1_target_velocity.z]
        current_velocity = [s.gripper1_velocity.x,
                            s.gripper1_velocity.y,
                            s.gripper1_velocity.z]
        ntv = np.linalg.norm(target_velocity)
        nv = np.linalg.norm(current_velocity)
        if abs(ntv - nv) > 0.155:
            at_constraint_boundary = True
        else:
            at_constraint_boundary = False

        if args.verbose:
            print(t, nv, ntv, at_constraint_boundary)

        combined_constraint_labels[t, 0] = at_constraint_boundary

        # publish the pull command
        action_msg.gripper1_pos.x = gripper1_target_x
        action_msg.gripper1_pos.y = gripper1_target_y
        action_msg.gripper2_pos.x = gripper2_target_x
        action_msg.gripper2_pos.y = gripper2_target_y
        services.action_pub.publish(action_msg)

        # let the simulator run
        step = WorldControlRequest()
        step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
        services.world_control(step)  # this will block until stepping is complete

    n_positive = np.count_nonzero(np.any(combined_constraint_labels, axis=1))
    percentage_positive = n_positive * 100.0 / combined_constraint_labels.shape[0]

    if args.verbose:
        print(Fore.GREEN + "Trajectory {} Complete".format(env_idx) + Fore.RESET)

    labels_dict = {
        LabelType.Combined.name: combined_constraint_labels,
    }
    data_dict = {
        'rope_configurations': rope_configurations,
        'gripper1_forces': gripper1_forces,
        'gripper1_velocities': gripper1_velocities,
        'gripper1_target_velocities': gripper1_target_velocities,
        'gripper2_forces': gripper2_forces,
        'gripper2_velocities': gripper2_velocities,
        'gripper2_target_velocities': gripper2_target_velocities,
        'images': images,
    }
    return data_dict, labels_dict, sdf_data, percentage_positive


def generate_trajs(args, full_output_directory, services):
    filename_pairs = []
    percentages_positive = []
    constraint_label_types = None
    n_trajs_per_file = 256

    states = np.ndarray((args.n_trajs, args.steps_per_traj, 2), np.float32)
    actions = np.ndarray((args.n_trajs, args.steps_per_traj, 2), np.float32)
    image_bytes = np.ndarray((args.n_trajs, args.steps_per_traj), object)

    for i in range(args.n_trajs):
        tf_record_idx = i / n_trajs_per_file
        current_record_traj_idx = i % n_trajs_per_file

        ####################
        # Move the objects #
        ####################
        # TODO: apply forces to the boxes

        ####################
        # Collect New Data #
        ####################
        data_dict, labels_dict, sdf_data, percentage_violation = generate_traj(args, services, i)

        head_positions = data_dict['rope_configurations'][:, 4:6]
        states[current_record_traj_idx] = head_positions
        actions[current_record_traj_idx] = data_dict['gripper1_target_velocities']
        image_bytes[current_record_traj_idx] = data_dict['images']

        constraint_label_types = list(labels_dict.keys())
        percentages_positive.append(percentage_violation)

        rope_data_filename = os.path.join(full_output_directory, 'rope_data_{:d}.npz'.format(i))
        sdf_filename = os.path.join(full_output_directory, 'sdf_data_{:d}.npz'.format(i))

        # FIXME: order matters
        filename_pairs.append([sdf_filename, rope_data_filename])

        data_dict.update(labels_dict)

        # Save the data
        np.savez(rope_data_filename, **data_dict)
        sdf_data.save(sdf_filename)

        if not args.verbose:
            print(".", end='')
            sys.stdout.flush()

        if current_record_traj_idx == 255:
            dataset = tf.data.Dataset.from_tensor_slices((image_bytes, states, actions))

            serialized_dataset = dataset.map(video_prediction_dataset_utils.tf_serialize_example)

            end_traj_idx = i
            start_traj_idx = end_traj_idx - n_trajs_per_file
            full_filename = "traj_{}_to_{}".format(start_traj_idx, end_traj_idx)
            writer = tf.data.experimental.TFRecordWriter(full_filename)
            print("saving {}".format(full_filename))
            writer.write(serialized_dataset)

    mean_percentage_positive = np.mean(percentages_positive)
    print("Class balance: mean % positive: {}".format(mean_percentage_positive))

    dataset_filename = os.path.join(full_output_directory, 'dataset.json')
    dataset = MultiEnvironmentDataset(filename_pairs, constraint_label_types=constraint_label_types,
                                      n_obstacles=-1, obstacle_size=-1, seed=args.seed)
    dataset.save(dataset_filename)


def generate(args):
    rospy.init_node('gazebo_small_with_images')

    full_output_directory = random_environment_data_utils.data_directory(args.outdir, args.n_trajs)
    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    if not args.seed:
        args.seed = np.random.randint(0, 10000)
    np.random.seed(args.seed)

    # fire up services
    services = GazeboServices()
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
    services.set_physics.call(set)

    # let the simulator run to get the first image
    step = WorldControlRequest()
    step.steps = int(0.02 / 0.001)  # assuming 0.001s per simulation step
    services.world_control(step)  # this will block until stepping is complete

    generate_trajs(args, full_output_directory, services)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=5000)

    parser = argparse.ArgumentParser()
    parser.add_argument("n_trajs", help='how many trajectories to collect', type=int)
    parser.add_argument("outdir")
    parser.add_argument('--res', '-r', type=float, default=0.01, help='size of cells in meters')
    parser.add_argument("--steps-per-traj", type=int, default=30)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--real-time-rate", help='number of times real time', type=float, default=10)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
