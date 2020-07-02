#!/usr/bin/env python
from __future__ import division, print_function

import time
import rospy
from typing import Dict, Optional, List

import numpy as np
from colorama import Fore
from ompl import base as ob

from link_bot_classifiers.rnn_recovery_model import RNNRecoveryModelWrapper
from link_bot_planning.my_planner import MyPlanner, MyPlannerStatus
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.ros_pycommon import get_occupancy_data


def get_environment_common(w_m: float, h_m: float, res: float, service_provider: BaseServices, scenario: ExperimentScenario):
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


def execute_actions(service_provider: BaseServices, scenario: ExperimentScenario, start_state: Dict, actions: List[Dict]):
    actual_path = [start_state]
    for action in actions:
        scenario.execute_action(action)
        state_t = scenario.get_state()
        actual_path.append(state_t)
    return actual_path


class PlanAndExecute:

    def __init__(self,
                 planner: MyPlanner,
                 n_total_plans: int,
                 n_plans_per_env: int,
                 verbose: int,
                 planner_params: Dict,
                 service_provider: BaseServices,
                 no_execution: bool,
                 seed: int,
                 recovery_actions_model: Optional[RNNRecoveryModelWrapper] = None,
                 pause_between_plans: Optional[bool] = False):
        self.pause_between_plans = pause_between_plans
        self.planner = planner
        self.recovery_actions_model = recovery_actions_model
        self.n_total_plans = n_total_plans
        self.n_plans_per_env = n_plans_per_env
        self.planner_params = planner_params
        self.verbose = verbose
        self.service_provider = service_provider
        self.no_execution = no_execution
        self.env_rng = np.random.RandomState(seed)
        self.goal_rng = np.random.RandomState(seed)

        self.total_plan_idx = 0
        self.n_failures = 0

    def run(self):
        self.total_plan_idx = 0
        while True:
            self.randomize_environment()
            for _ in range(self.n_plans_per_env):
                success = self.plan_and_execute_once()
                if success:
                    self.total_plan_idx += 1
                    if self.total_plan_idx >= self.n_total_plans:
                        self.on_complete()
                        return

    def plan_and_execute_once(self):
        # get start states
        start_state = self.planner.scenario.get_state()

        # get the environment, which here means anything which is assumed constant during planning
        # This includes the occupancy map but can also include things like the initial state of the tether

        environment = get_environment_for_extents_3d(extent=self.planner_params['extent'],
                                                     res=self.planner.classifier_model.data_collection_params['res'],
                                                     service_provider=self.service_provider,
                                                     robot_name=self.planner.fwd_model.scenario.robot_name())

        # generate a random target
        goal = self.planner.scenario.sample_goal(self.planner_params['goal_extent'], rng=self.goal_rng)

        if self.verbose >= 1:
            (Fore.MAGENTA + "Planning to {}".format(goal) + Fore.RESET)

        ############
        # Planning #
        ############
        t0 = time.time()
        planner_result = self.planner.plan(start_state, environment, goal)
        planner_data = ob.PlannerData(self.planner.si)
        self.planner.planner.getPlannerData(planner_data)

        if self.verbose >= 1:
            rospy.loginfo(planner_result.planner_status)

        self.on_after_plan()

        if planner_result.planner_status in [MyPlannerStatus.Failure, MyPlannerStatus.NotProgressing]:
            if self.recovery_actions_model is not None:
                print("performing recovery action!")
                current_state = scenario.get_state()
                recovery_actions = self.recovery_actions_model.sample(environment, current_state)
                self.execute_actions(start_state, recovery_actions)
            return False
        elif planner_result.planner_status in [MyPlannerStatus.Solved, MyPlannerStatus.Timeout]:
            planning_time = time.time() - t0
            if self.verbose >= 1:
                rospy.loginfo(f"Planning time: {planning_time:5.3f}s")

            self.on_plan_complete(planner_result.path, goal, planner_result.actions, environment, planner_data,
                                  planning_time, planner_result.planner_status)

            # execute the plan, collecting the states that actually occurred
            if not self.no_execution:
                if self.verbose >= 2:
                    print(Fore.CYAN + "Executing Plan" + Fore.RESET)

                actual_path = self.execute_actions(start_state, planner_result.actions)
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
            return True
        else:
            raise NotImplementedError()

    def on_plan_complete(self,
                         planned_path: List[Dict],
                         goal,
                         planned_actions: List[Dict],
                         environment: Dict,
                         planner_data: ob.PlannerData,
                         planning_time: float,
                         planner_status: MyPlannerStatus):
        # visualize the plan
        if self.verbose >= 1:
            self.planner.scenario.animate_final_path(environment, planned_path, planned_actions)

    def on_execution_complete(self,
                              planned_path: List[Dict],
                              planned_actions: List[Dict],
                              goal,
                              actual_path: List[Dict],
                              environment: Dict,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: MyPlannerStatus):
        pass

    def on_complete(self):
        pass

    def on_planner_failure(self,
                           start_states: Dict[str, np.ndarray],
                           goal,
                           environment: Dict,
                           planner_data: ob.PlannerData):
        pass

    def on_after_plan(self):
        pass

    def randomize_environment(self):
        rospy.logerr("Randomizing environment in planning not implemented!")

    def execute_actions(self, start_state: Dict, actions: Optional[List[Dict]]) -> List[Dict]:
        return execute_actions(self.service_provider, self.planner.scenario, start_state, actions)
