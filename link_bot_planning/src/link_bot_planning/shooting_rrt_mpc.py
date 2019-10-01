#!/usr/bin/env python
from __future__ import division, print_function

import time

import numpy as np
import rospy
import tensorflow as tf
from colorama import Fore

from link_bot_data import random_environment_data_utils
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import get_sdf_data
from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_gazebo.srv import LinkBotStateRequest, LinkBotTrajectoryRequest, WorldControlRequest
from link_bot_planning.goals import sample_goal
from link_bot_planning.shooting_rrt import ShootingRRT

tf.enable_eager_execution()


def plan_and_execute_random_goals(args, fwd_model, dt):
    rospy.init_node('collect_classifier_data')

    ##############################################
    #             NN Constraint Model            #
    ##############################################
    constraint_tf_graph = tf.Graph()
    with constraint_tf_graph.as_default():

        class ConstraintCheckerWrapper:

            def __init__(self):
                pass

            @staticmethod
            def get_graph():
                return constraint_tf_graph

            def __call__(self, np_state):
                p_reject = 0.0
                return p_reject

    rrt = ShootingRRT(fwd_model=fwd_model,
                      constraint_checker_wrapper=ConstraintCheckerWrapper(),
                      dt=dt,
                      max_v=args.max_v,
                      n_state=fwd_model.n_state,
                      planner_timeout=args.planner_timeout,
                      env_w=args.env_w,
                      env_h=args.env_h)

    services = gazebo_utils.setup_gazebo_env(args.verbose, args.real_time_rate, reset_world=False)
    # wait for obstacles to settle
    step = WorldControlRequest()
    step.steps = 5000
    services.world_control(step)
    full_sdf_data = get_sdf_data(args.env_h, args.env_w, args.res, services)

    # Statistics
    n_fails = 0
    n_successes = 0
    execution_times = []
    planning_times = []
    min_true_costs = []

    for trial_idx in range(args.n_trials):
        # generate a random goal
        state_req = LinkBotStateRequest()
        state = services.get_state(state_req)
        head_idx = state.link_names.index("head")
        start_configuration = gazebo_utils.points_to_config(state.points)
        start_head_point = state.points[head_idx]
        tail_goal = sample_goal(args.env_w, args.env_h, start_head_point, env_padding=0.1)
        start = np.expand_dims(np.array(start_configuration), axis=0)
        goal_tail_point = np.array(tail_goal)  # a 2d point for the tail. This will be converted to a goal region in the RRT

        random_environment_data_utils.publish_markers(args,
                                                      goal_tail_point[0], goal_tail_point[1],
                                                      start_configuration[0], start_configuration[1],
                                                      marker_size=0.05)

        planning_start_time = time.time()
        planned_actions, planned_path, _ = rrt.plan(start, goal_tail_point, full_sdf_data.sdf, args.verbose)
        planning_time = time.time() - planning_start_time

        traj_req = LinkBotTrajectoryRequest()
        traj_req.dt = dt

        for action in planned_actions[:args.n_actions]:
            action_msg = LinkBotVelocityAction()
            action_msg.gripper1_velocity.x = action[0]
            action_msg.gripper1_velocity.y = action[1]
            traj_req.gripper1_traj.append(action_msg)

        # execute the plan, collecting the states that actually occurred
        if args.verbose >= 2:
            print(Fore.CYAN + "Executing Plan.".format(goal_tail_point) + Fore.RESET)

        execution_start_time = time.time()
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

        final_distance = np.linalg.norm(actual_rope_configurations[-1, 0:2] - goal_tail_point)
        min_distance = np.min(np.linalg.norm(actual_rope_configurations[:, 0:2] - goal_tail_point))

        if final_distance < args.success_threshold:
            n_successes += 1
        else:
            n_fails += 1

        execution_time = time.time() - execution_start_time
        execution_times.append(execution_time)
        planning_times.append(planning_time)
        min_true_costs.append(min_distance)

    return np.array(min_true_costs), np.array(execution_times), np.array(planning_times), n_fails, n_successes
