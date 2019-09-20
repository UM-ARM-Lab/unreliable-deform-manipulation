#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import os
import pathlib

import numpy as np
import ompl.util as ou
import rospy
import tensorflow as tf
from colorama import Fore
from geometry_msgs.msg import Vector3

from link_bot_data import random_environment_data_utils
from link_bot_gaussian_process import link_bot_gp
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import get_sdf_data
from link_bot_gazebo.srv import LinkBotPathRequest, LinkBotStateRequest
from link_bot_planning import gp_rrt
from link_bot_planning.goals import sample_goal


def collect_classifier_data(args):
    rospy.init_node('collect_classifier_data')

    fwd_gp_model = link_bot_gp.LinkBotGP()
    fwd_gp_model.load(os.path.join(args.gp_model_dir, 'fwd_model'))
    gp_model_path_info = args.gp_model_dir.parts[1:]

    dt = fwd_gp_model.dataset_hparams['dt']

    assert args.env_w >= args.sdf_w
    assert args.env_h >= args.sdf_h

    full_output_directory = random_environment_data_utils.data_directory(args.outdir, *gp_model_path_info)
    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'dt': dt,
            'sdf_w': args.sdf_w,
            'sdf_h': args.sdf_h,
            'env_w': args.env_w,
            'env_h': args.env_h,
        }
        json.dump(options, of, indent=1)

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
        print("Using seed: ", args.seed)
    np.random.seed(args.seed)

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

    rrt = gp_rrt.GPRRT(fwd_gp_model=fwd_gp_model,
                       inv_gp_model=None,
                       constraint_checker_wrapper=ConstraintCheckerWrapper(),
                       dt=dt,
                       max_v=args.max_v,
                       planner_timeout=args.planner_timeout)

    services = gazebo_utils.setup_gazebo_env(args.verbose, args.real_time_rate)
    # generate a new environment by rearranging the obstacles
    for traj_idx in range(args.n_envs):
        # generate a bunch of plans to random goals
        state_req = LinkBotStateRequest()

        # Center SDF at the current head position
        state = services.get_state(state_req)
        head_idx = state.link_names.index("head")
        initial_head_point = np.array([state.points[head_idx].x, state.points[head_idx].y])

        # Compute SDF Data
        local_sdf, local_sdf_gradient, local_sdf_origin, sdf_data = get_sdf_data(args, initial_head_point, services)

        for plan_idx in range(args.n_targets_per_env):
            # generate a random target
            state = services.get_state(state_req)
            head_idx = state.link_names.index("head")
            rope_configuration = gazebo_utils.points_to_config(state.points)
            head_point = state.points[head_idx]
            goal = sample_goal(args.env_w, args.env_h, head_point)

            # plan to that target
            planned_path, _, _ = rrt.plan(rope_configuration, goal, sdf_data.sdf, args.verbose)

            path_req = LinkBotPathRequest()
            for np_point in planned_path:
                point = Vector3()
                point.x = np_point[0]
                point.y = np_point[1]
                point.z = np_point[2]
                path_req.gripper1_path.append(point)

            # execute the plan, collecting the states that actually occurred
            path_res = services.execute_path(planned_path, services, args, dt)

            # collect the transition pairs (s_t, s_{t+1}, \hat{s}_t, \hat{s}_{t+1}) and  the corresponding label
            # The label will be based on thresholding the euclidian distance ||s_{t+1} - \hat{s}_{t+1}||2
            # If the distance is greater than the threshold, then the label is 0 to indicate the model should not be used there.

            # save to a TF record


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("gp_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--n-targets-per-env", type=int, default=2)
    parser.add_argument("--seed", '-s', type=int, default=1)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=60.0)
    parser.add_argument("--real-time-rate", type=float, default=1.0)
    parser.add_argument('--res', '-r', type=float, default=0.01, help='size of cells in meters')
    parser.add_argument('--env-w', type=float, default=1.0)
    parser.add_argument('--env-h', type=float, default=1.0)
    parser.add_argument('--sdf-w', type=float, default=1.0)
    parser.add_argument('--sdf-h', type=float, default=1.0)
    parser.add_argument('--max-v', type=float, default=0.15)

    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_DEBUG)
    # ou.setLogLevel(ou.LOG_ERROR)

    collect_classifier_data(args)


if __name__ == '__main__':
    main()
