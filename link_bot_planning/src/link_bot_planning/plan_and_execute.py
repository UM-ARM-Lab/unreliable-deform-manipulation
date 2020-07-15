#!/usr/bin/env python
import time
import pathlib
import rospy
from typing import Dict, Optional, List

import numpy as np
from colorama import Fore
from ompl import base as ob

from link_bot_planning.my_planner import MyPlanner, MyPlannerStatus, PlanningResult
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from link_bot_pycommon.base_services import BaseServices
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy
from link_bot_classifiers.random_recovery_policy import RandomRecoveryPolicy
from link_bot_classifiers import recovery_policy_utils
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.ros_pycommon import get_occupancy_data


def execute_actions(service_provider: BaseServices,
                    scenario: ExperimentScenario,
                    start_state: Dict,
                    actions: List[Dict],
                    plot: bool = False):
    pre_action_state = start_state
    actual_path = [pre_action_state]
    for action in actions:
        scenario.execute_action(action)
        state_t = scenario.get_state()
        actual_path.append(state_t)
        if plot:
            scenario.plot_executed_action(pre_action_state, action)
            scenario.plot_state_rviz(state_t, label='actual')
        pre_action_state = state_t
    return actual_path


class PlanAndExecute:

    def __init__(self,
                 planner: MyPlanner,
                 n_plans: int,
                 n_plans_per_env: int,
                 verbose: int,
                 planner_params: Dict,
                 service_provider: BaseServices,
                 no_execution: bool,
                 seed: int,
                 pause_between_plans: Optional[bool] = False):
        self.pause_between_plans = pause_between_plans
        self.planner = planner
        self.n_plans = n_plans
        self.n_plans_per_env = n_plans_per_env
        self.planner_params = planner_params
        self.verbose = verbose
        self.service_provider = service_provider
        self.no_execution = no_execution
        self.env_rng = np.random.RandomState(seed)
        self.goal_rng = np.random.RandomState(seed)
        if self.planner_params['recovery']['use_recovery']:
            recovery_model_dir = pathlib.Path(self.planner_params['recovery']['recovery_model_dir'])
            self.recovery_policy = recovery_policy_utils.load_generic_model(model_dir=recovery_model_dir,
                                                                            scenario=self.planner.scenario,
                                                                            rng=np.random.RandomState(seed))
        else:
            self.recovery_policy = None

        self.plan_idx = 0
        self.n_failures = 0

    def run(self):
        self.plan_idx = 0
        while True:
            for _ in range(self.n_plans_per_env):
                done = self.run_and_check_valid()
                if done:
                    return
                self.randomize_environment()

    def run_and_check_valid(self):
        if self.planner_params['recovery']['use_recovery']:
            run_was_valid = self.plan_and_execute_with_recovery()
        else:
            run_was_valid = self.plan_and_execute_without_recovery()
        if run_was_valid:
            # only count if it was valid
            self.plan_idx += 1
            if self.plan_idx >= self.n_plans:
                self.on_complete()
                return True
        return False

    def setup_planning_query(self):
        # get start states
        start_state = self.planner.scenario.get_state()

        # get the environment, which here means anything which is assumed constant during planning
        # This includes the occupancy map but can also include things like the initial state of the tether
        environment = get_environment_for_extents_3d(extent=self.planner_params['extent'],
                                                     res=self.planner.classifier_model.data_collection_params['res'],
                                                     service_provider=self.service_provider,
                                                     robot_name=self.planner.fwd_model.scenario.robot_name())

        # Get the goal (default is to randomly sample one)
        goal = self.get_goal(environment)

        planning_query_info = {
            'goal': goal,
            'environment': environment,
            'start_state': start_state,
        }
        return planning_query_info

    def plan_with_random_restarts_when_not_progressing(self, planning_query_info: Dict):
        for _ in range(self.planner_params['n_random_restarts'] + 1):
            # retry on "Failure" or "Not Progressing"
            planning_result = self.planner.plan(environment=planning_query_info['environment'],
                                                start_state=planning_query_info['start_state'],
                                                goal=planning_query_info['goal'])
            if planning_result.status == MyPlannerStatus.Solved:
                break
            if planning_result.status == MyPlannerStatus.Timeout:
                break
        return planning_result

    def plan(self, planning_query_info: Dict):
        ############
        # Planning #
        ############
        if self.verbose >= 1:
            (Fore.MAGENTA + "Planning to {}".format(planning_query_info['goal']) + Fore.RESET)
        planning_result = self.plan_with_random_restarts_when_not_progressing(planning_query_info)
        rospy.loginfo(f"Planning time: {planning_result.time:5.3f}s, Status: {planning_result.status}")

        self.on_plan_complete(planning_query_info, planning_result)

        return planning_result

    def execute(self, planning_query_info: Dict, planning_result: PlanningResult):
        # execute the plan, collecting the states that actually occurred
        self.on_before_execute()
        if self.no_execution:
            state_t = self.planner.scenario.get_state()
            actual_path = [state_t]
        else:
            if self.verbose >= 2:
                print(Fore.CYAN + "Executing Plan" + Fore.RESET)
            start_state = planning_query_info['start_state']
            plot = self.verbose >= 1
            actual_path = execute_actions(self.service_provider,
                                          self.planner.scenario,
                                          start_state,
                                          planning_result.actions,
                                          plot=plot)
        # post-execution callback
        execution_result = {
            'path': actual_path
        }
        return execution_result

    def execute_recovery_action(self, action: Dict):
        if self.no_execution:
            pass
        else:
            self.planner.scenario.execute_action(action)

    def plan_and_execute_with_recovery(self):
        n_attempts = self.planner_params['recovery']['n_attempts']
        recovery_actions_taken = []
        for attempt_idx in range(n_attempts):
            planning_query_info = self.setup_planning_query()

            planning_result = self.plan(planning_query_info)

            if planning_result.status == MyPlannerStatus.Failure:
                # this run won't count if we return false, the environment will be randomized, then we'll try again
                return False
            elif planning_result.status == MyPlannerStatus.NotProgressing:
                recovery_action = self.recovery_policy(environment=planning_query_info['environment'],
                                                       state=planning_query_info['start_state'])
                if self.verbose >= 1:
                    # +1 to make it more human friendly
                    rospy.loginfo(f"Attempting recovery action {attempt_idx + 1} of {n_attempts}")

                if self.verbose >= 3:
                    rospy.loginfo("Chosen Recovery Action:")
                    rospy.loginfo(recovery_action)
                recovery_actions_taken.append(recovery_action)
                self.execute_recovery_action(recovery_action)
            else:
                if self.verbose >= 2 and attempt_idx > 0:
                    rospy.loginfo(f"recovery succeeded on attempt {attempt_idx}")
                break
        recovery_actions_result = {
            'attempt_idx': attempt_idx,
            'recovery_actions_taken': recovery_actions_taken,
        }

        execution_result = self.execute(planning_query_info, planning_result)
        self.on_execution_complete(planning_query_info, planning_result, execution_result, recovery_actions_result)

        return True

    # FIXME: don't need this special case probably
    def plan_and_execute_without_recovery(self):
        planning_query_info = self.setup_planning_query()

        planning_result = self.plan(planning_query_info)

        if planning_result.status == MyPlannerStatus.Failure:
            # this run won't count if we return false, the environment will be randomized, then we'll try again
            return False

        execution_result = self.execute(planning_query_info, planning_result)
        recovery_actions_result = {
            'attempt_idx': 0,
            'recovery_actions_taken': [],
        }
        self.on_execution_complete(planning_query_info, planning_result, execution_result, recovery_actions_result)

        return True

    def get_goal(self, environment: Dict):
        goal = self.planner.scenario.sample_goal(environment=environment,
                                                 rng=self.goal_rng,
                                                 planner_params=self.planner_params)
        return goal

    def on_plan_complete(self,
                         planning_query_info: Dict,
                         planning_result: PlanningResult):
        # visualize the plan
        if self.verbose >= 1:
            self.planner.scenario.animate_final_path(environment=planning_query_info['environment'],
                                                     planned_path=planning_result.path,
                                                     actions=planning_result.actions)

    def on_before_execute(self):
        pass

    def on_execution_complete(self,
                              planning_query_info: Dict,
                              planning_result: PlanningResult,
                              execution_result: Dict,
                              recovery_actions_result: Dict):
        pass

    def on_complete(self):
        pass

    def randomize_environment(self):
        self.planner.scenario.randomize_environment(self.env_rng, self.planner_params, self.planner_params)
