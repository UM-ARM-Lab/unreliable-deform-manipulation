#!/usr/bin/env python
from __future__ import division, print_function

import time
from typing import Dict, Optional, List

import numpy as np
from colorama import Fore
from ompl import base as ob

from link_bot_planning import my_planner
from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_planning.goals import sample_collision_free_goal
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import SimParams
from link_bot_pycommon.base_services import Services
from link_bot_pycommon.ros_pycommon import get_occupancy_data
from link_bot_pycommon.ros_pycommon import get_states_dict
from peter_msgs.msg import Action


def get_environment_common(w_m: float, h_m: float, res: float, service_provider: Services, scenario: ExperimentScenario):
    full_env_data = get_occupancy_data(env_w_m=w_m,
                                       env_h_m=h_m,
                                       res=res,
                                       service_provider=service_provider,
                                       robot_name=scenario.robot_name())
    environment = {
        'full_env/env': full_env_data.data,
        'full_env/origin': full_env_data.origin,
        'full_env/res': full_env_data.resolution,
        'full_env/extent': full_env_data.extent,
    }
    return environment


def execute_plan(service_provider, dt, actions):
    start_states = get_states_dict(service_provider)
    actual_path = [start_states]
    for t in range(actions.shape[0]):
        action_request = Action()
        action_request.max_time_per_step = dt
        action_request.action = actions[t]
        action_response = service_provider.execute_action(action_request)
        state_t = {}
        for named_object in action_response.objects.objects:
            state_t[named_object.name] = np.array(named_object.state_vector)
        actual_path.append(state_t)
    return actual_path


class PlanAndExecute:

    def __init__(self,
                 planner: MyPlanner,
                 n_total_plans: int,
                 n_plans_per_env: int,
                 verbose: int,
                 planner_params: Dict,
                 sim_params: SimParams,
                 service_provider: Services,
                 no_execution: bool,
                 seed: int,
                 retry_on_failure: Optional[bool] = True,
                 pause_between_plans: Optional[bool] = False):
        self.pause_between_plans = pause_between_plans
        self.retry_on_failure = retry_on_failure
        self.planner = planner
        self.n_total_plans = n_total_plans
        self.n_plans_per_env = n_plans_per_env
        self.sim_params = sim_params
        self.planner_params = planner_params
        self.verbose = verbose
        self.service_provider = service_provider
        self.no_execution = no_execution
        self.env_rng = np.random.RandomState(seed)
        self.goal_rng = np.random.RandomState(seed)

        # remove all markers
        self.service_provider.marker_provider.remove_all()

        self.plan_idx = 0
        self.total_plan_idx = 0

    def run(self):
        self.total_plan_idx = 0
        initial_poses_in_collision = 0
        while True:
            self.on_before_plan()

            self.plan_idx = 0
            while True:
                # get start states
                start_states = get_states_dict(self.service_provider)

                # get the environment, which here means anything which is assumed constant during planning
                # This includes the occupancy map but can also include things like the initial state of the tether

                environment = get_environment_common(w_m=self.planner.full_env_params.w,
                                                     h_m=self.planner.full_env_params.h,
                                                     res=self.planner.full_env_params.res,
                                                     service_provider=self.service_provider,
                                                     scenario=self.planner.scenario)

                environment.update(self.planner.scenario.get_environment_from_start_states_dict(start_states))

                # generate a random target
                goal = self.get_goal(self.planner_params['goal_w_m'], self.planner_params['goal_h_m'], environment)

                if self.verbose >= 1:
                    # publish goal marker
                    self.planner.scenario.publish_goal_marker(self.service_provider.marker_provider,
                                                              goal,
                                                              self.planner_params['goal_threshold'])

                if self.verbose >= 1:
                    print(Fore.CYAN + "Planning from {} to {}".format(start_states, goal) + Fore.RESET)

                t0 = time.time()
                planner_result = self.planner.plan(start_states, environment, goal)
                my_planner.interpret_planner_status(planner_result.planner_status, self.verbose)
                planner_data = ob.PlannerData(self.planner.si)
                self.planner.planner.getPlannerData(planner_data)

                if self.verbose >= 1:
                    print(planner_result.planner_status.asString())

                self.on_after_plan()

                if not planner_result.planner_status:
                    print("fail!")
                    self.on_planner_failure(start_states, goal, environment, planner_data)
                    if self.retry_on_failure:
                        break
                else:  # Approximate or Exact solution found!
                    planning_time = time.time() - t0
                    if self.verbose >= 1:
                        print("Planning time: {:5.3f}s".format(planning_time))

                    self.on_plan_complete(planner_result.path, goal, planner_result.actions, environment, planner_data,
                                          planning_time, planner_result.planner_status)

                    # execute the plan, collecting the states that actually occurred
                    if not self.no_execution:
                        if self.verbose >= 2:
                            print(Fore.CYAN + "Executing Plan.".format(goal) + Fore.RESET)

                        actual_path = self.execute_plan(planner_result.actions)
                        self.on_execution_complete(planner_result.path,
                                                   planner_result.actions,
                                                   goal,
                                                   actual_path,
                                                   environment,
                                                   planner_data,
                                                   planning_time,
                                                   planner_result.planner_status)

                    if self.pause_between_plans:
                        input("Press enter to proceed to next plan...")

                self.plan_idx += 1
                self.total_plan_idx += 1
                if self.plan_idx >= self.n_plans_per_env or self.total_plan_idx >= self.n_total_plans:
                    break

            if self.total_plan_idx >= self.n_total_plans:
                break

        self.on_complete(initial_poses_in_collision)

    def get_goal(self, w_meters, h_meters, environment):
        return sample_collision_free_goal(goal_w_m=w_meters, goal_h_m=h_meters, environment=environment, rng=self.goal_rng)

    def on_plan_complete(self,
                         planned_path: List[Dict],
                         goal,
                         planned_actions: np.ndarray,
                         environment: Dict,
                         planner_data: ob.PlannerData,
                         planning_time: float,
                         planner_status: ob.PlannerStatus):
        pass

    def on_execution_complete(self,
                              planned_path: List[Dict],
                              planned_actions: np.ndarray,
                              goal,
                              actual_path: List[Dict],
                              environment: Dict,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: ob.PlannerStatus):
        pass

    def on_complete(self, initial_poses_in_collision):
        pass

    def on_planner_failure(self,
                           start_states: Dict[str, np.ndarray],
                           goal,
                           environment: Dict,
                           planner_data: ob.PlannerData):
        pass

    def on_after_plan(self):
        pass

    def on_before_plan(self):
        if self.sim_params.nudge is not None:
            self.service_provider.nudge(self.planner.n_action)

        if self.sim_params.movable_obstacles is not None:
            # FIXME: instead of hard coding obstacles names, use the /objects service
            # generate a new environment by rearranging the obstacles
            self.service_provider.move_objects(self.sim_params.max_step_size,
                                               self.sim_params.movable_obstacles,
                                               self.planner.full_env_params.w,
                                               self.planner.full_env_params.h,
                                               padding=0,
                                               rng=self.env_rng)

    def execute_plan(self, actions):
        """
        :param actions: currently a numpy array, [time, n_action]
        :return: the states, a list of Dicts
        """
        return execute_plan(self.service_provider, self.planner.fwd_model.dt, actions)
