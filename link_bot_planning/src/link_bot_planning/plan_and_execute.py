#!/usr/bin/env python
import pathlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np
import rospy
import tensorflow as tf
from colorama import Fore
from dataclasses_json import dataclass_json

from link_bot_classifiers import recovery_policy_utils
from link_bot_planning.my_planner import MyPlanner, MyPlannerStatus, PlanningResult, PlanningQuery
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import extent_to_bbox
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d


class TrialStatus(Enum):
    Reached = "reached"
    Timeout = "timeout"
    NotProgressingNoRecovery = "not_progressing_no_recovery"


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
                 trials: List[int],
                 verbose: int,
                 planner_params: Dict,
                 service_provider: BaseServices,
                 no_execution: bool):
        self.planner = planner
        self.trials = trials
        self.planner_params = planner_params
        self.verbose = verbose
        self.service_provider = service_provider
        self.no_execution = no_execution
        self.env_rng = np.random.RandomState(0)
        self.goal_rng = np.random.RandomState(0)
        self.recovery_rng = np.random.RandomState(0)
        if self.planner_params['recovery']['use_recovery']:
            recovery_model_dir = pathlib.Path(self.planner_params['recovery']['recovery_model_dir'])
            self.recovery_policy = recovery_policy_utils.load_generic_model(model_dir=recovery_model_dir,
                                                                            scenario=self.planner.scenario,
                                                                            rng=self.recovery_rng)
        else:
            self.recovery_policy = None

        self.n_failures = 0

        # # Debugging
        # self.goal_bbox_pub = rospy.Publisher('goal_bbox', BoundingBox, queue_size=10, latch=True)
        # bbox_msg = extent_to_bbox(planner_params['goal_extent'])
        # bbox_msg.header.frame_id = 'world'
        # self.goal_bbox_pub.publish(bbox_msg)

    def run(self):
        self.planner.scenario.randomization_initialization()
        for trial_idx in self.trials:
            self.env_rng.seed(trial_idx)
            self.recovery_rng.seed(trial_idx)
            self.goal_rng.seed(trial_idx)
            # NOTE: ompl SetSeed can only be called once which is why we don't bother doing it here
            # FIXME: we should not be relying on this...
            np.random.seed(trial_idx)
            tf.random.set_seed(trial_idx)

            self.randomize_environment()

            self.plan_and_execute(trial_idx)

        self.on_complete()

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

    def plan_and_execute(self, trial_idx: int):
        self.on_start_trial(trial_idx)

        start_time = time.perf_counter()
        total_timeout = self.planner_params['total_timeout']

        # Get the goal (default is to randomly sample one)
        goal = self.get_goal(self.get_environment())

        attempt_idx = 0
        steps_data = []
        planning_queries = []
        while True:
            # get start states
            start_state = self.planner.scenario.get_state()

            # get the environment, which here means anything which is assumed constant during planning
            # This includes the occupancy map but can also include things like the initial state of the tether
            environment = self.get_environment()

            planning_query = PlanningQuery(goal=goal, environment=environment, start=start_state, seed=trial_idx)
            planning_queries.append(planning_query)

            planning_result = self.plan(planning_query)

            time_since_start = time.perf_counter() - start_time

            if planning_result.status == MyPlannerStatus.Failure:
                # this run won't count if we return false, the environment will be randomized, then we'll try again
                raise RuntimeError("planning failed -- is the start state out of bounds?")
            elif planning_result.status == MyPlannerStatus.NotProgressing:
                if self.recovery_policy is None:
                    # Nothing else to do here, just give up
                    end_state = self.planner.scenario.get_state()
                    trial_status = TrialStatus.NotProgressingNoRecovery
                    print(
                        Fore.BLUE + f"Trial {trial_idx} Ended: not progressing, no recovery. {time_since_start:.3f}s" + Fore.RESET)
                    trial_data_dict = {
                        'planning_queries': planning_queries,
                        'total_time': time_since_start,
                        'trial_status': trial_status,
                        'trial_idx': trial_idx,
                        'end_state': end_state,
                        'goal': goal,
                        'steps': steps_data,
                    }
                    self.on_trial_complete(trial_data_dict, trial_idx)
                    return
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
                    print(Fore.BLUE + f"Trial {trial_idx} Ended: Goal reached!" + Fore.RESET)
                else:
                    trial_status = TrialStatus.Timeout
                    print(Fore.BLUE + f"Trial {trial_idx} Ended: Timeout {time_since_start:.3f}s" + Fore.RESET)
                trial_data_dict = {
                    'planning_queries': planning_queries,
                    'total_time': time_since_start,
                    'trial_status': trial_status,
                    'trial_idx': trial_idx,
                    'goal': goal,
                    'steps': steps_data,
                    'end_state': end_state,
                }
                self.on_trial_complete(trial_data_dict, trial_idx)
                return

    def on_trial_complete(self, trial_data, trial_idx: int):
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

    def on_start_trial(self, trial_idx: int):
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
