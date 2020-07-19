#!/usr/bin/env python
import time
import pathlib
import rospy
from dataclasses import dataclass
from typing import Dict, Optional, List

from enum import Enum
import numpy as np
from colorama import Fore
from ompl import base as ob
from dataclasses_json import dataclass_json

from link_bot_planning.my_planner import MyPlanner, MyPlannerStatus, PlanningResult, PlanningQuery
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from link_bot_pycommon.base_services import BaseServices
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy
from moonshine.moonshine_utils import listify
from link_bot_classifiers.random_recovery_policy import RandomRecoveryPolicy
from link_bot_classifiers import recovery_policy_utils
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.ros_pycommon import get_occupancy_data


class TrialStatus(Enum):
    Reached = "reached"
    Timeout = "timeout"


@dataclass_json
@dataclass
class ExecutionResult:
    path: List[Dict]


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
                 n_trials: int,
                 verbose: int,
                 planner_params: Dict,
                 service_provider: BaseServices,
                 no_execution: bool,
                 seed: int):
        self.planner = planner
        self.n_trials = n_trials
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

        self.trial_idx = 0
        self.n_failures = 0

    def run(self):
        self.trial_idx = 0
        attempt_idx = 0
        while True:
            done = self.run_and_check_valid()
            if done:
                return
            self.randomize_environment()
            attempt_idx += 1

    def run_and_check_valid(self):
        run_was_valid = self.plan_and_execute()
        if run_was_valid:
            # only count if it was valid
            self.trial_idx += 1
            if self.trial_idx >= self.n_trials:
                self.on_complete()
                return True
        return False

    def setup_planning_query(self):
        # get start states
        start_state = self.planner.scenario.get_state()

        # get the environment, which here means anything which is assumed constant during planning
        # This includes the occupancy map but can also include things like the initial state of the tether
        environment = self.get_environment()

        # Get the goal (default is to randomly sample one)
        goal = self.get_goal(environment)

        planning_query = PlanningQuery(goal=goal, environment=environment, start_state=start_state)
        return planning_query

    def plan(self, planning_query: Dict):
        ############
        # Planning #
        ############
        if self.verbose >= 1:
            (Fore.MAGENTA + "Planning to {}".format(planning_query.goal) + Fore.RESET)
        planning_result = self.planner.plan(planning_query=planning_query)
        rospy.loginfo(f"Planning time: {planning_result.time:5.3f}s, Status: {planning_result.status}")

        self.on_plan_complete(planning_query, planning_result)

        return planning_result

    def execute(self, planning_query: Dict, planning_result: PlanningResult):
        # execute the plan, collecting the states that actually occurred
        self.on_before_execute()
        if self.no_execution:
            state_t = self.planner.scenario.get_state()
            actual_path = [state_t]
        else:
            if self.verbose >= 2 and not self.no_execution:
                print(Fore.CYAN + "Executing Plan" + Fore.RESET)
            actual_path = execute_actions(self.service_provider,
                                          self.planner.scenario,
                                          planning_query.start,
                                          planning_result.actions,
                                          plot=True)
        # post-execution callback
        execution_result = ExecutionResult(path=actual_path)
        return execution_result

    def execute_recovery_action(self, action: Dict):
        if self.no_execution:
            actual_path = []
        else:
            before_state = self.planner.scenario.get_state()
            self.planner.scenario.execute_action(action)
            after_state = self.planner.scenario.get_state()
            actual_path = [before_state, after_state]
        execution_result = ExecutionResult(path=actual_path)
        return execution_result

    def get_environment(self):
        # get the environment, which here means anything which is assumed constant during planning
        # This includes the occupancy map but can also include things like the initial state of the tether
        return get_environment_for_extents_3d(extent=self.planner_params['extent'],
                                              res=self.planner.classifier_model.data_collection_params['res'],
                                              service_provider=self.service_provider,
                                              robot_name=self.planner.fwd_model.scenario.robot_name())

    def plan_and_execute(self):
        start_time = time.perf_counter()
        total_timeout = self.planner_params['total_timeout']

        # Get the goal (default is to randomly sample one)
        goal = self.get_goal(self.get_environment())

        attempt_idx = 0
        steps_data = []
        while True:
            # get start states
            start_state = self.planner.scenario.get_state()

            # get the environment, which here means anything which is assumed constant during planning
            # This includes the occupancy map but can also include things like the initial state of the tether
            environment = self.get_environment()

            planning_query = PlanningQuery(goal=goal, environment=environment, start=start_state)

            planning_result = self.plan(planning_query)

            time_since_start = time.perf_counter() - start_time

            if planning_result.status == MyPlannerStatus.Failure:
                # this run won't count if we return false, the environment will be randomized, then we'll try again
                return False
            elif planning_result.status == MyPlannerStatus.NotProgressing:
                if self.recovery_policy is None:
                    pass  # do nothing
                else:
                    recovery_action = self.recovery_policy(environment=planning_query.environment,
                                                           state=planning_query.start)
                    attempt_idx += 1
                    rospy.loginfo(f"Attempting recovery action {attempt_idx}")

                    if self.verbose >= 3:
                        rospy.loginfo("Chosen Recovery Action:")
                        rospy.loginfo(recovery_action)
                    execution_result = self.execute_recovery_action(recovery_action)
                    # Extract planner data now before it goes out of scope (in C++)
                    steps_data.append({
                        'type': 'executed_recovery',
                        'planning_query': planning_query,
                        'planning_result': planning_result,
                        'recovery_action': recovery_action,
                        'execution_result': execution_result,
                        'time_since_start': time_since_start,
                    })
            else:
                execution_result = self.execute(planning_query, planning_result)
                steps_data.append({
                    'type': 'executed_plan',
                    'planning_query': planning_query,
                    'planning_result': planning_result,
                    'execution_result': execution_result,
                    'time_since_start': time_since_start,
                })
                self.on_execution_complete(planning_query, planning_result, execution_result)

            end_state = self.planner.scenario.get_state()
            d = self.planner.scenario.distance_to_goal(end_state, planning_query.goal)
            rospy.loginfo(f"distance to goal after execution is {d:.3f}")
            reached_goal = (d <= self.planner_params['goal_threshold'] + 1e-6)

            if reached_goal or time_since_start > total_timeout:
                if reached_goal:
                    trial_status = TrialStatus.Reached
                    print(Fore.BLUE + f"Trial {self.trial_idx} Ended: Goal reached!" + Fore.RESET)
                else:
                    trial_status = TrialStatus.Timeout
                    print(Fore.BLUE + f"Trial {self.trial_idx} Ended: Timeout {time_since_start:.3f}" + Fore.RESET)
                trial_data_dict = {
                    'total_time': time_since_start,
                    'trial_status': trial_status,
                    'trial_idx': self.trial_idx,
                    'goal': goal,
                    'steps': steps_data,
                }
                self.on_trial_complete(trial_data_dict)
                break
        return True

    def on_trial_complete(self, trial_data):
        pass

    def get_goal(self, environment: Dict):
        goal = self.planner.scenario.sample_goal(environment=environment,
                                                 rng=self.goal_rng,
                                                 planner_params=self.planner_params)
        return goal

    def on_plan_complete(self,
                         planning_query: Dict,
                         planning_result: PlanningResult):
        # visualize the plan
        if self.verbose >= 1:
            self.planner.scenario.animate_final_path(environment=planning_query.environment,
                                                     planned_path=planning_result.path,
                                                     actions=planning_result.actions)

    def on_before_execute(self):
        pass

    def on_execution_complete(self,
                              planning_query: Dict,
                              planning_result: PlanningResult,
                              execution_result: Dict):
        pass

    def on_complete(self):
        pass

    def randomize_environment(self):
        self.planner.scenario.randomize_environment(self.env_rng, self.planner_params, self.planner_params)
