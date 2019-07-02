#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import rosnode
import rospy
from colorama import Fore

from gazebo_msgs.srv import SpawnModelRequest
from ignition import markers
from link_bot_gazebo.msg import LinkBotConfiguration, MultiLinkBotPositionAction
from link_bot_gazebo.srv import ComputeSDF
from link_bot_gazebo.srv import WorldControl, WorldControlRequest, LinkBotState, LinkBotStateRequest
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import link_bot_sdf_tools

DT = 0.1  # seconds per time step


def random_yaw():
    return np.random.uniform(-np.pi, np.pi)


def norm_dot(force, velocity):
    return np.dot(velocity, force) / (np.linalg.norm(velocity) + 1e-9) * np.linalg.norm(force)
    # return np.dot(speed, force) / (np.linalg.norm(force) + 1e-9)
    # return np.dot(force, speed) / (np.linalg.norm(force) * np.linalg.norm(speed) + 1e-9)


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


def make_gazebo_world_file(args, rope_length, rope_x, rope_y):
    world_xml_root = ET.fromstring("""<?xml version="1.0" ?>
        <sdf version="1.5">
            <world name="default">
                <gui>
                    <camera name='user_camera'>
                        <pose frame=''>0 0 25 0 1.5707 1.5707</pose>
                    </camera>
                </gui>
                <include>
                    <uri>model://ground_plane</uri>
                </include>
                <include>
                    <uri>model://sun</uri>
                </include>
                <include>
                    <uri>model://arena_5</uri>
                    <pose>0 0 0 0 0 0</pose>
                    <name>arena</name>
                </include>
                <plugin name="stepping_plugin" filename="libstepping_plugin.so"/>
                <plugin name="collision_map_plugin" filename="libcollision_map_plugin.so"/>
                <physics name="ode" type="ode">
                    <real_time_update_rate>10000</real_time_update_rate>
                    <ode>
                        <solver>
                          <type>quick</type>
                        </solver>
                    </ode>
                </physics>
            </world>
        </sdf>
        """)
    world_element = world_xml_root.find('world')
    with open("/tmp/temp.world", 'w+') as world_file:

        # Pick random locations to place obstacles
        for i in range(args.n_obstacles):
            obstacle_msg = SpawnModelRequest()
            name = "box_{}".format(i)
            obstacle_msg.model_name = name
            box_size = args.obstacle_size * args.res * 2
            while True:
                box_x = np.random.uniform(-args.w / 2, args.w / 2)
                box_y = np.random.uniform(-args.h / 2, args.h / 2)
                tl1_x = box_x - box_size / 2
                tl1_y = box_y + box_size / 2
                br1_x = box_x + box_size / 2
                br1_y = box_y - box_size / 2
                tl2_x = rope_x - rope_length
                tl2_y = rope_y + rope_length
                br2_x = rope_x + rope_length
                br2_y = rope_y - rope_length
                if (tl1_x > br2_x or tl2_x > br1_x) or (tl1_y < br2_y or tl2_y < br1_y):
                    break

            obstacle_msg.initial_pose.position.x = box_x
            obstacle_msg.initial_pose.position.y = box_y
            obstacle_msg.initial_pose.position.z = 0
            box_element = ET.fromstring(
                "<include><name>box_{}</name><uri>model://box</uri><pose>{} {} 0 0 0 0</pose></include>".format(i, box_x, box_y))
            world_element.append(box_element)

        # Launch the gazebo world
        tree = ET.ElementTree(world_xml_root)
        tree.write(world_file)
    return world_file


def init_simulation(args, run_idx, env_idx, cli_args, init_config):
    if args.verbose:
        stdout_file = sys.stdout
        stderr_file = sys.stderr
    else:
        stdout_file = open(".roslaunch.stdout-{}.{}".format(run_idx, env_idx), 'w')
        stderr_file = open(".roslaunch.stdout-{}.{}".format(run_idx, env_idx), 'w')

    process = subprocess.Popen(cli_args, stdout=stdout_file, stderr=stderr_file)  # something long running
    sleep(2.0)

    # wait until gazebo is up
    while not rosnode.rosnode_ping("gazebo", max_count=1):
        pass

    # fire up the services
    if args.verbose:
        print(Fore.CYAN + "Waiting for services..." + Fore.RESET)

    rospy.wait_for_service("/world_control")
    rospy.wait_for_service("/link_bot_state")
    rospy.wait_for_service("/sdf")

    if args.verbose:
        print(Fore.CYAN + "Done waiting for services" + Fore.RESET)

    action_pub = rospy.Publisher("/multi_link_bot_position_action", MultiLinkBotPositionAction, queue_size=10)
    config_pub = rospy.Publisher('/link_bot_configuration', LinkBotConfiguration, queue_size=10, latch=True)
    world_control = rospy.ServiceProxy('/world_control', WorldControl)
    get_state = rospy.ServiceProxy('/link_bot_state', LinkBotState)
    compute_sdf = rospy.ServiceProxy('/sdf', ComputeSDF)

    # set the rope configuration
    config_pub.publish(init_config)

    # let the simulator run
    step = WorldControlRequest()
    step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
    world_control(step)  # this will block until stepping is complete

    return process, (action_pub, world_control, get_state, compute_sdf)


def publish_markers(args, target_x, target_y, rope_x, rope_y):
    target_marker = markers.make_marker(rgb=[1, 0, 0], id=1)
    target_marker.pose.position.x = target_x
    target_marker.pose.position.y = target_y
    rope_marker = markers.make_marker(rgb=[0, 1, 0], id=2)
    rope_marker.pose.position.x = rope_x
    rope_marker.pose.position.y = rope_y
    if not args.headless:
        markers.publish(target_marker)
        markers.publish(rope_marker)


def generate_env(args, env_idx):
    # place rope at a random location
    rope_length = 1.5
    n_joints = 8
    init_config = LinkBotConfiguration()
    rope_x = np.random.uniform(-args.w / 2 + rope_length, args.w / 2 - rope_length)
    rope_y = np.random.uniform(-args.h / 2 + rope_length, args.h / 2 - rope_length)
    init_config.tail_pose.x = rope_x
    init_config.tail_pose.y = rope_y
    init_config.tail_pose.theta = random_yaw()
    init_config.joint_angles_rad = np.clip(np.random.randn(n_joints) * np.pi / 4, -np.pi / 2, np.pi / 2)

    # construct gazebo world
    world_file = make_gazebo_world_file(args, rope_length, rope_x, rope_y)

    cli_args = ['roslaunch',
                'link_bot_gazebo',
                'multi_link_bot.launch',
                'world_name:={}'.format(world_file.name),
                'verbose:={}'.format(bool(args.verbose)),
                'gui:={}'.format(not args.headless),
                ]

    run_idx = 0
    state_req = LinkBotStateRequest()
    action_msg = MultiLinkBotPositionAction()
    while True:
        process, services = init_simulation(args, run_idx, env_idx, cli_args, init_config)
        action_pub, world_control, get_state, compute_sdf = services

        # Compute SDF Data
        sdf_data = link_bot_sdf_tools.request_sdf_data(compute_sdf, res=args.res)

        # Create random rope configurations by picking a random point and applying forces to move the rope to that point
        rope_configurations = np.ndarray((args.steps, 6), dtype=np.float32)
        gripper_forces = np.ndarray((args.steps, 4))
        gripper_velocities = np.ndarray((args.steps, 4))
        constraint_labels = np.ndarray((args.steps, 2), dtype=np.float32)
        target_x = np.random.uniform(-args.w / 2, args.w / 2)
        target_y = np.random.uniform(-args.h / 2, args.h / 2)

        publish_markers(args, target_x, target_y, rope_x, rope_y)

        history_size = 5
        gripper1_velocity_history = np.zeros((history_size, 3))
        gripper1_force_history = np.zeros((history_size, 3))
        for t in range(args.steps):
            # save the state and action data
            link_bot_state = get_state(state_req)
            rope_configurations[t] = np.array([link_bot_state.tail_x,
                                               link_bot_state.tail_y,
                                               link_bot_state.mid_x,
                                               link_bot_state.mid_y,
                                               link_bot_state.head_x,
                                               link_bot_state.head_y])
            gripper_forces[t] = np.array([link_bot_state.gripper1_force.x,
                                          link_bot_state.gripper1_force.y,
                                          link_bot_state.gripper2_force.x,
                                          link_bot_state.gripper2_force.y])
            gripper_velocities[t] = np.array([link_bot_state.gripper1_velocity.x,
                                              link_bot_state.gripper1_velocity.y,
                                              link_bot_state.gripper2_velocity.x,
                                              link_bot_state.gripper2_velocity.y])

            # TODO: use ground truth labels not just based on force/velocity?
            # Use a simple median filter to check whether we are at the constraint boundary
            # check if the angle of the vectors of force and vector of velocity are close enough to 0

            gripper1_velocity_vec = np.array([link_bot_state.gripper1_velocity.x, link_bot_state.gripper1_velocity.y, 0])
            gripper1_force_vec = np.array([link_bot_state.gripper1_force.x, link_bot_state.gripper1_force.y, 0])
            gripper1_force_history = np.roll(gripper1_force_history, 1, axis=0)
            gripper1_force_history[0] = gripper1_force_vec
            gripper1_velocity_history = np.roll(gripper1_velocity_history, 1, axis=0)
            gripper1_velocity_history[0] = gripper1_velocity_vec

            filtered_gripper1_force = np.median(gripper1_force_history, axis=0)
            filtered_gripper1_velocity = np.median(gripper1_velocity_history, axis=0)
            normalized_dot = norm_dot(filtered_gripper1_force, filtered_gripper1_velocity)

            if normalized_dot > -10000 or t < history_size:
                at_constraint_boundary = False
            else:
                at_constraint_boundary = True

            if args.verbose:
                print(normalized_dot, at_constraint_boundary)

            constraint_labels[t, 0] = at_constraint_boundary
            constraint_labels[t, 1] = at_constraint_boundary

            # publish the pull command
            action_msg.gripper1_pos.x = target_x
            action_msg.gripper1_pos.y = target_y
            action_pub.publish(action_msg)

            # let the simulator run
            step = WorldControlRequest()
            step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
            world_control(step)  # this will block until stepping is complete

        n_positive = np.count_nonzero(np.any(constraint_labels, axis=1))
        percentage_positive = n_positive * 100.0 / constraint_labels.shape[0]

        return_status = process.poll()
        print(return_status)
        if return_status is None or return_status == 0:
            # everything is fine, so end the experiment
            if args.verbose:
                print(Fore.GREEN + "Trial {} Complete".format(env_idx) + Fore.RESET)
            process.terminate()
            break
        else:
            print(Fore.RED + "Process died! Restarting" + Fore.RESET)
        if not args.retry:
            if args.verbose:
                print(Fore.GREEN + "Trial {} Complete".format(env_idx) + Fore.RESET)
            break

        all_nodes = rosnode.rosnode_listnodes()
        rosnode.kill_nodes(all_nodes)
        sleep(1.0)
        process.wait()

    return (rope_configurations, gripper_forces, gripper_velocities, constraint_labels), sdf_data, percentage_positive


def generate(args):
    rospy.init_node('apply_random_steps')

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
    for env_idx in range(args.envs):
        data, sdf_data, percentage_violation = generate_env(args, env_idx)
        rope_configurations, gripper_forces, gripper_velocities, constraint_labels = data
        percentages_positive.append(percentage_violation)
        if args.outdir:
            rope_data_filename = os.path.join(full_output_directory, 'rope_data_{:d}.npz'.format(env_idx))
            sdf_filename = os.path.join(full_output_directory, 'sdf_data_{:d}.npz'.format(env_idx))

            # FIXME: order matters
            filename_pairs.append([sdf_filename, rope_data_filename])

            np.savez(rope_data_filename,
                     rope_configurations=rope_configurations,
                     gripper_forces=gripper_forces,
                     gripper_velocities=gripper_velocities,
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


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("steps", help='how many steps to do', type=int)
    parser.add_argument("envs", help='how many environments to generate', type=int)
    parser.add_argument('w', type=int, help='environment with in meters (int)')
    parser.add_argument('h', type=int, help='environment with in meters (int)')
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
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--retry", action="store_true")

    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
