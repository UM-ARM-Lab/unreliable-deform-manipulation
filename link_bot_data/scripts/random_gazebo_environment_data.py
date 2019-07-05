#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from time import sleep

import numpy as np
import psutil
import rosnode
import rospy
from colorama import Fore

from gazebo_msgs.srv import SpawnModelRequest
from ignition import markers
from link_bot_data import random_environment_data_utils
from link_bot_gazebo.msg import LinkBotConfiguration, MultiLinkBotPositionAction
from link_bot_gazebo.srv import ComputeSDF
from link_bot_gazebo.srv import WorldControl, WorldControlRequest, LinkBotState, LinkBotStateRequest
from link_bot_models.label_types import LabelType
from link_bot_pycommon import link_bot_sdf_tools

DT = 0.1  # seconds per time step


def random_yaw():
    return np.random.uniform(-np.pi, np.pi)


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
                    <real_time_update_rate>{}</real_time_update_rate>
                    <ode>
                        <solver>
                            <type>quick</type>
                        </solver>
                        <constraints>
                            <contact_surface_layer>0.01</contact_surface_layer>
                        </constraints>
                    </ode>
                </physics>
            </world>
        </sdf>
        """.format(args.real_time_rate * 1000))
    world_element = world_xml_root.find('world')
    box_locations = []
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
                """
                <model name="box_{0}">
                    <static>true</static>
                    <link name="link_1">
                      <pose>{3} {4} {2} 0 0 0</pose>
                      <visual name="visual">
                        <geometry>
                          <box>
                            <size>{1} {1} {1}</size>
                          </box>
                        </geometry>
                      </visual>
                      <collision name="box_collision">
                        <geometry>
                          <box>
                            <size>{1} {1} {1}</size>
                          </box>
                        </geometry>
                      </collision>
                    </link>
                  </model>
                """.format(i, box_size, box_size / 2, box_x, box_y))

            world_element.append(box_element)
            box_locations.append([box_x, box_y])

        # Launch the gazebo world
        tree = ET.ElementTree(world_xml_root)
        tree.write(world_file)
    return world_file, box_locations


def init_simulation(args, run_idx, env_idx, cli_args, init_config):
    if args.verbose:
        stdout_file = sys.stdout
        stderr_file = sys.stderr
    else:
        stdout_file = open(".roslaunch.stdout-{}.{}".format(run_idx, env_idx), 'w')
        stderr_file = open(".roslaunch.stdout-{}.{}".format(run_idx, env_idx), 'w')

    process = subprocess.Popen(cli_args, stdout=stdout_file, stderr=stderr_file)

    # fire up the services
    if args.verbose or (env_idx == 0 and run_idx == 0):
        print(Fore.CYAN + "Waiting for services..." + Fore.RESET)

    rospy.wait_for_service("/world_control")
    rospy.wait_for_service("/link_bot_state")
    rospy.wait_for_service("/sdf")

    if args.verbose or (env_idx == 0 and run_idx == 0):
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
    markers.publish(target_marker)
    markers.publish(rope_marker)


def extract_from_state(link_bot_state):
    rope_configurations = np.array([link_bot_state.tail_x,
                                    link_bot_state.tail_y,
                                    link_bot_state.mid_x,
                                    link_bot_state.mid_y,
                                    link_bot_state.head_x,
                                    link_bot_state.head_y])
    gripper_forces = np.array([link_bot_state.gripper1_force.x,
                               link_bot_state.gripper1_force.y,
                               link_bot_state.gripper2_force.x,
                               link_bot_state.gripper2_force.y])
    gripper_velocities = np.array([link_bot_state.gripper1_velocity.x,
                                   link_bot_state.gripper1_velocity.y,
                                   link_bot_state.gripper2_velocity.x,
                                   link_bot_state.gripper2_velocity.y])
    return rope_configurations, gripper_forces, gripper_velocities


def generate_env(args, env_idx):
    # place rope at a random location
    rope_length = 1.05
    n_joints = 8
    init_config = LinkBotConfiguration()
    wall_thickness = 0.05
    rope_x = np.random.uniform(-args.w / 2 + rope_length + wall_thickness, args.w / 2 - rope_length - wall_thickness)
    rope_y = np.random.uniform(-args.h / 2 + rope_length + wall_thickness, args.h / 2 - rope_length - wall_thickness)
    init_config.tail_pose.x = rope_x
    init_config.tail_pose.y = rope_y
    init_config.tail_pose.theta = random_yaw()
    init_config.joint_angles_rad = np.clip(np.random.randn(n_joints) * np.pi / 4, -np.pi / 2, np.pi / 2)

    # construct gazebo world
    world_file, box_locations = make_gazebo_world_file(args, rope_length, rope_x, rope_y)

    cli_args = ['roslaunch',
                'link_bot_gazebo',
                'multi_link_bot.launch',
                'world_name:={}'.format(world_file.name),
                'verbose:={}'.format(bool(args.verbose)),
                'gui:={}'.format(not args.headless),
                'spawn_x:={}'.format(rope_x),
                'spawn_y:={}'.format(rope_y),
                ]

    run_idx = 0
    state_req = LinkBotStateRequest()
    action_msg = MultiLinkBotPositionAction()
    process, services = init_simulation(args, run_idx, env_idx, cli_args, init_config)
    action_pub, world_control, get_state, compute_sdf = services

    # Compute SDF Data
    sdf_data = link_bot_sdf_tools.request_sdf_data(compute_sdf, width=args.w, height=args.h, res=args.res)

    # Create random rope configurations by picking a random point and applying forces to move the rope to that point
    rope_configurations = np.ndarray((args.steps, 6), dtype=np.float32)
    gripper_forces = np.ndarray((args.steps, 4))
    gripper_velocities = np.ndarray((args.steps, 4))
    combined_constraint_labels = np.ndarray((args.steps, 1), dtype=np.float32)

    target_x = 0
    target_y = 0
    for t in range(args.steps):
        # save the state and action data
        link_bot_state = get_state(state_req)
        rope_configurations[t], gripper_forces[t], gripper_velocities[t] = extract_from_state(link_bot_state)

        # TODO: use ground truth labels not just based on force/velocity?
        target_velocity = [link_bot_state.gripper1_target_velocity.x,
                           link_bot_state.gripper1_target_velocity.y,
                           link_bot_state.gripper1_target_velocity.z]
        current_velocity = [link_bot_state.gripper1_velocity.x,
                            link_bot_state.gripper1_velocity.y,
                            link_bot_state.gripper1_velocity.z]
        ntv = np.linalg.norm(target_velocity)
        nv = np.linalg.norm(current_velocity)
        if abs(ntv - nv) > 0.1:
            at_constraint_boundary = True
        else:
            at_constraint_boundary = False

        if args.verbose:
            print(nv, ntv, at_constraint_boundary)

        combined_constraint_labels[t, 0] = at_constraint_boundary

        if t % args.new_goal_period == 0:
            # target_x = np.random.uniform(-args.w / 2, args.w / 2)
            # target_y = np.random.uniform(-args.h / 2, args.h / 2)
            box_idx = np.random.choice(len(box_locations))
            target_x, target_y = box_locations[box_idx]
            if not args.headless:
                publish_markers(args, target_x, target_y, rope_x, rope_y)

        # publish the pull command
        action_msg.gripper1_pos.x = target_x
        action_msg.gripper1_pos.y = target_y
        action_pub.publish(action_msg)

        # let the simulator run
        step = WorldControlRequest()
        step.steps = int(DT / 0.001)  # assuming 0.001s per simulation step
        world_control(step)  # this will block until stepping is complete

    n_positive = np.count_nonzero(np.any(combined_constraint_labels, axis=1))
    percentage_positive = n_positive * 100.0 / combined_constraint_labels.shape[0]

    if args.verbose:
        print(Fore.GREEN + "Trial {} Complete".format(env_idx) + Fore.RESET)

    rosnode.kill_nodes(["/gazebo"])
    process.terminate()
    process.wait()

    if args.verbose:
        # allow possible error text to finish printing
        sleep(1.0)

    labels_dict = {
        LabelType.Combined: combined_constraint_labels,
    }
    return rope_configurations, labels_dict, sdf_data, percentage_positive


def generate(args):
    # first kill old gazebo processes
    gz_running = True
    while gz_running:
        for proc in psutil.process_iter():
            executable = proc.cmdline()[0]
            if "gzserver" in proc.name() or "gzclient" in proc.name() or "gzserver" in executable or "gzclient" in executable:
                print("killing old gazebo instance!")
                proc.kill()
                break
        else:
            print("done")
            gz_running = False

    rospy.init_node('random_gazebo_environment_data')

    full_output_directory = random_environment_data_utils.data_directory(args.outdir, args.envs, args.steps)

    if not args.seed:
        # I know this looks crazy, but the idea is that when we run the script multiple times we don't want to get the same output
        # but we als do want to be able to recreate the output from a seed, so we generate a random seed if non is provided
        args.seed = np.random.randint(0, 10000)
    np.random.seed(args.seed)

    # Define what kinds of labels are contained in this dataset
    random_environment_data_utils.generate_envs(args, full_output_directory, generate_env)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("steps", help='how many steps to do', type=int)
    parser.add_argument("envs", help='how many environments to generate', type=int)
    parser.add_argument('w', type=int, help='environment with in meters (int)')
    parser.add_argument('h', type=int, help='environment with in meters (int)')
    parser.add_argument("--outdir", help='directory dataset will go in')
    parser.add_argument('--res', '-r', type=float, default=0.05, help='size of cells in meters')
    parser.add_argument('--n-obstacles', type=int, default=20, help='size of obstacles in cells')
    parser.add_argument('--obstacle-size', type=int, default=5, help='size of obstacles in cells')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("-Q", help="dimensions in constraint checking output space", type=int, default=1)
    parser.add_argument("--save-frequency", '-f', help='save every this many steps', type=int, default=10)
    parser.add_argument("--new-goal-period", help='change target rope position every this many time steps', type=int, default=100)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--real-time-rate", help='number of times real time', type=float, default=10)
    parser.add_argument("--verbose", '-v', action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--retry", action="store_true")

    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
