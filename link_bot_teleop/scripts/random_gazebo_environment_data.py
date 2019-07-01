#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import sys
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import rospy
from colorama import Fore

from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, DeleteModel, DeleteModelRequest
from link_bot_gazebo.msg import LinkBotConfiguration, LinkBotVelocityAction
from link_bot_gazebo.srv import WorldControl, WorldControlRequest, LinkBotState, LinkBotStateRequest
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import link_bot_pycommon, link_bot_sdf_tools
from sdf_tools.srv import ComputeSDF

DT = 0.1  # seconds per time step


def random_yaw():
    return np.random.uniform(-np.pi, np.pi)


def plot(args, sdf_data, threshold, rope_configuration, constraint_labels):
    del args  # unused
    plt.figure()
    binary = sdf_data.sdf < threshold
    plt.imshow(np.flipud(binary.T), extent=sdf_data.extent)

    xs = [rope_configuration[0], rope_configuration[2], rope_configuration[4]]
    ys = [rope_configuration[1], rope_configuration[3], rope_configuration[5]]
    sdf_constraint_color = 'r' if constraint_labels[0] else 'g'
    overstretched_constraint_color = 'r' if constraint_labels[1] else 'g'
    plt.plot(xs, ys, linewidth=0.5, zorder=1, c=overstretched_constraint_color)
    plt.scatter(rope_configuration[4], rope_configuration[5], s=16, c=sdf_constraint_color, zorder=2)

    plt.figure()
    plt.imshow(np.flipud(sdf_data.sdf.T), extent=sdf_data.extent)
    subsample = 2
    x_range = np.arange(sdf_data.extent[0], sdf_data.extent[1], subsample * sdf_data.resolution[0])
    y_range = np.arange(sdf_data.extent[0], sdf_data.extent[1], subsample * sdf_data.resolution[1])
    y, x = np.meshgrid(y_range, x_range)
    dx = sdf_data.gradient[::subsample, ::subsample, 0]
    dy = sdf_data.gradient[::subsample, ::subsample, 1]
    plt.quiver(x, y, dx, dy, units='x', scale=10)


def generate_env(args, config_pub, world_control, get_state, action_pub, spawn_model, delete_model, compute_sdf_service):
    # teleport to the origin
    init_config = LinkBotConfiguration()
    init_config.tail_pose.x = 0
    init_config.tail_pose.y = 0
    init_config.tail_pose.theta = 0
    init_config.joint_angles_rad = [0, 0, 0, 0, 0, 0, 0, 0]
    config_pub.publish(init_config)

    # let the world step once
    step = WorldControlRequest()
    step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
    world_control.call(step)  # this will block until stepping is complete

    sleep(0.5)

    state_req = LinkBotStateRequest()
    action_msg = LinkBotVelocityAction()
    action_msg.control_link_name = 'head'

    # Delete the previous obstacles
    for i in range(args.n_obstacles):
        name = "box_{}".format(i)
        delete_request = DeleteModelRequest()
        delete_request.model_name = name
        delete_response = delete_model.call(delete_request)
        if not delete_response.success:
            print("failed to delete model {}".format(name))

    # Pick random locations to place obstacles
    for i in range(args.n_obstacles):
        obstacle_msg = SpawnModelRequest()
        name = "box_{}".format(i)
        obstacle_msg.model_name = name
        obstacle_msg.model_xml = """
        <?xml version="1.0" ?>
        <sdf version="1.5">
          <model name="{1}">
            <static>true</static>
            <link name="link_1">
              <pose>0 0 0.5 0 0 0</pose>
              <visual name="visual">
                <geometry>
                  <box>
                    <size>{0} {0} {0}</size>
                  </box>
                </geometry>
              </visual>
              <collision name="box_collision">
                <geometry>
                  <box>
                    <size>{0} {0} {0}</size>
                  </box>
                </geometry>
              </collision>
            </link>
          </model>
        </sdf>
        """.format(args.obstacle_size * args.res, name)
        obstacle_msg.initial_pose.position.x = np.random.uniform(-5, 5)
        obstacle_msg.initial_pose.position.y = np.random.uniform(-5, 5)
        spawn_model.call(obstacle_msg)

    # Compute SDF Data
    sdf_data = link_bot_sdf_tools.request_sdf_data(compute_sdf_service, res=args.res)

    # Create random rope configurations by picking a random location and applying forces to move the rope in that direction
    rope_configurations = np.ndarray((args.steps, 6), dtype=np.float32)
    torques = np.ndarray((args.steps, args.steps + 1, args.N))
    constraint_labels = np.ndarray((args.steps, 2), dtype=np.float32)
    head_vx = 1
    head_vy = 0
    for t in range(args.steps):
        # save the state and action data
        link_bot_state = get_state.call(state_req)
        rope_configurations[t] = np.array([link_bot_state.tail_x,
                                           link_bot_state.tail_y,
                                           link_bot_state.mid_x,
                                           link_bot_state.mid_y,
                                           link_bot_state.head_x,
                                           link_bot_state.head_y])
        torques[t] = np.array([link_bot_state.tail_torque.x,
                               link_bot_state.tail_torque.y,
                               link_bot_state.mid_torque.x,
                               link_bot_state.mid_torque.y,
                               link_bot_state.head_torque.x,
                               link_bot_state.head_torque.y])
        constraint_labels[t, 0] = link_bot_state.head_in_contact
        constraint_labels[t, 1] = link_bot_state.overstretched

        # publish the pull command
        action_msg.vx = head_vx
        action_msg.vy = head_vy
        action_pub.publish(action_msg)

        # let the simulator run
        step = WorldControlRequest()
        step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
        world_control.call(step)  # this will block until stepping is complete

    n_positive = np.count_nonzero(np.any(constraint_labels, axis=1))
    percentage_positive = n_positive * 100.0 / constraint_labels.shape[0]

    return rope_configurations, constraint_labels, sdf_data, percentage_positive


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("envs", help='how many environments to generate', type=int)
    parser.add_argument("steps", help='how many steps to do', type=int)
    parser.add_argument("--outdir", help='directory dataset will go in')
    parser.add_argument('--res', '-r', type=float, default=0.05, help='size of cells in meters')
    parser.add_argument('--n-obstacles', type=int, default=14, help='size of obstacles in cells')
    parser.add_argument('--obstacle-size', type=int, default=8, help='size of obstacles in cells')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("-Q", help="dimensions in constraint checking output space", type=int, default=1)
    parser.add_argument("--save-frequency", '-f', help='save every this many steps', type=int, default=10)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    rospy.init_node('apply_random_steps')

    action_pub = rospy.Publisher("/link_bot_velocity_action", LinkBotVelocityAction, queue_size=10)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    get_state = rospy.ServiceProxy('/link_bot_state', LinkBotState)
    compute_sdf_service = rospy.ServiceProxy('/sdf', ComputeSDF)
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

    ############################
    full_output_directory = '{}_{}_{}'.format(args.outdir, args.envs, args.steps)
    if args.outdir:
        if os.path.isfile(full_output_directory):
            print(Fore.RED + "argument outdir is an existing file, aborting." + Fore.RESET)
            return
        elif not os.path.isdir(full_output_directory):
            os.mkdir(full_output_directory)

    if not args.seed:
        # I know this looks crazy, but the idea is that when we run the script multiple times we don't want to get the same output
        # but we als do want to be able to recreate the output from a seed, so we generate a random seed if non is provided
        args.seed = np.random.randint(0, 10000)
    np.random.seed(args.seed)

    # Define what kinds of labels are contained in this dataset
    constraint_label_types = [LabelType.SDF, LabelType.Overstretching]

    filename_pairs = []
    percentages_positive = []
    for i in range(args.envs):
        rope_configurations, constraint_labels, sdf_data, percentage_violation = generate_env(args, config_pub, world_control,
                                                                                              get_state, action_pub, spawn_model,
                                                                                              delete_model, compute_sdf_service)
        percentages_positive.append(percentage_violation)
        if args.outdir:
            rope_data_filename = os.path.join(full_output_directory, 'rope_data_{:d}.npz'.format(i))
            sdf_filename = os.path.join(full_output_directory, 'sdf_data_{:d}.npz'.format(i))

            # FIXME: order matters
            filename_pairs.append([sdf_filename, rope_data_filename])

            np.savez(rope_data_filename,
                     rope_configurations=rope_configurations,
                     constraints=constraint_labels)
            sdf_data.save(sdf_filename)
        print(".", end='')
        sys.stdout.flush()
    print("done")

    mean_percentage_positive = np.mean(percentages_positive)
    print("Class balance: mean % positive: {}".format(mean_percentage_positive))

    if args.outdir:
        dataset_filename = os.path.join(full_output_directory, 'dataset.json')
        dataset = MultiEnvironmentDataset(filename_pairs, constraint_label_types=constraint_label_types,
                                          n_obstacles=args.n_obstacles, obstacle_size=args.obstacle_size,
                                          threshold=None, seed=args.seed)
        dataset.save(dataset_filename)


if __name__ == '__main__':
    main()
