import gzip
import json
import pathlib
from typing import Optional, Dict, List

import rospy
import numpy as np
import ompl.util as ou
import tensorflow as tf
from colorama import Fore
from ompl import base as ob
from time import perf_counter
from victor import victor_services
from moonshine.moonshine_utils import listify
from link_bot_pycommon.base_services import BaseServices
from link_bot_gazebo_python import gazebo_services
from link_bot_planning.ompl_viz import planner_data_to_json
from link_bot_planning.my_planner import MyPlanner, MyPlannerStatus
from link_bot_planning.get_planner import get_planner
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_planning import plan_and_execute


def dummy_proof_write(data, filename):
    t0 = perf_counter()
    while True:
        try:
            with gzip.open(filename, 'wb') as data_file:
                data_str = json.dumps(data)
                data_file.write(data_str.encode("utf-8"))
            return
        except KeyboardInterrupt:
            pass


class EvalPlannerConfigs(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 service_provider: BaseServices,
                 planner_config_name: str,
                 n_plans_per_env: int,
                 n_plans: int,
                 verbose: int,
                 planner_params: Dict,
                 comparison_item_idx: int,
                 seed: int,
                 goal,
                 outdir: pathlib.Path,
                 record: Optional[bool] = False,
                 pause_between_plans: Optional[bool] = False,
                 no_execution: Optional[bool] = False,
                 ):
        super().__init__(planner,
                         n_plans=n_plans,
                         n_plans_per_env=n_plans_per_env,
                         verbose=verbose,
                         planner_params=planner_params,
                         service_provider=service_provider,
                         pause_between_plans=pause_between_plans,
                         no_execution=no_execution,
                         seed=seed)
        self.record = record
        self.planner_config_name = planner_config_name
        self.outdir = outdir
        self.seed = seed

        self.subfolder = "{}_{}".format(self.planner_config_name, comparison_item_idx)
        self.root = self.outdir / self.subfolder
        self.root.mkdir(parents=True)
        print(Fore.CYAN + str(self.root) + Fore.RESET)
        self.failures_root = self.root / 'failures'
        self.successfully_completed_plan_idx = 0
        self.goal = goal

        metadata = {
            "n_plans": self.n_plans,
            "n_plans_per_env": self.n_plans_per_env,
            "planner_params": self.planner_params,
            "scenario": self.planner.scenario.simple_name(),
            "seed": self.seed,
            "horizon": self.planner.classifier_model.horizon,
        }
        with (self.root / 'metadata.json').open("w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    def randomize_environment(self):
        super().randomize_environment()

    def on_after_plan(self):
        if self.record:
            filename = self.root.absolute() / 'plan-{}.avi'.format(self.plan_idx)
            self.service_provider.start_record_trial(str(filename))

        super().on_after_plan()

    def get_goal(self, environment: Dict):
        if self.goal is not None:
            if self.verbose >= 1:
                print("Using Goal {}".format(self.goal))
            return self.goal
        else:
            return super().get_goal()

    def on_execution_complete(self,
                              planned_path: List[Dict],
                              planned_actions: np.ndarray,
                              goal,
                              actual_path: List[Dict],
                              environment: Dict,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: MyPlannerStatus):
        num_nodes = planner_data.numVertices()

        final_planned_state = planned_path[-1]
        plan_to_goal_error = self.planner.scenario.distance_to_goal(final_planned_state, goal)

        final_state = actual_path[-1]
        execution_to_goal_error = self.planner.scenario.distance_to_goal(final_state, goal)

        plan_to_execution_error = self.planner.scenario.distance(final_state, final_planned_state)

        print("{}: {}".format(self.subfolder, self.successfully_completed_plan_idx))

        planned_path_listified = listify(planned_path)
        planned_actions_listified = listify(planned_actions)
        actual_path_listified = listify(actual_path)
        tree_json = planner_data_to_json(planner_data, self.planner.scenario)

        data_for_plan = {
            "n_plans": self.n_plans,
            "n_targets": self.n_plans_per_env,
            "planner_params": self.planner_params,
            "scenario": self.planner.scenario.simple_name(),
            "seed": self.seed,
            'planner_status': planner_status.value,
            'environment': listify(environment),
            'planned_path': planned_path_listified,
            'actions': planned_actions_listified,
            'actual_path': actual_path_listified,
            'planning_time': planning_time,
            'plan_to_goal_error': plan_to_goal_error,
            'execution_to_goal_error': float(execution_to_goal_error),
            'plan_to_execution_error': float(plan_to_execution_error),
            'tree_json': tree_json,
            'goal': listify(goal),
            'num_nodes': num_nodes
        }
        data_filename = self.root / f'{self.successfully_completed_plan_idx}_metrics.json.gz'
        dummy_proof_write(data_for_plan, data_filename)

        self.successfully_completed_plan_idx += 1

        if self.record:
            self.service_provider.stop_record_trial()

    def on_planner_failure(self, start_states, tail_goal_point, environment: Dict, planner_data):
        info = {
            'start_states': {k: v.tolist() for k, v in start_states.items()},
            'tail_goal_point': tail_goal_point,
            'sdf': {
                'res': environment['res'],
                'origin': environment['origin'].tolist(),
                'extent': environment['extent'].tolist(),
                'data': environment['env'].tolist(),
            },
        }
        info_filename = self.root / "failures" / f'{self.n_failures}_info.json.gz'
        dummy_proof_write(info, info_filename)


def evaluate_planning_method(args, comparison_idx, planner_params, p_params_name, common_output_directory):
    # start at the same seed every time to make the planning environments & plans the same (hopefully?)
    # setting OMPL random seed should have no effect, because I use numpy's random in my sampler?
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)  # not sure if this has any effect

    # override some arguments
    if args.timeout is not None:
        rospy.loginfo(f"Overriding with timeout {args.timeout}")
        planner_params["termination_criteria"]['timeout'] = args.timeout

    planner_config_name = p_params_name.stem

    # Start Services
    if args.env_type == 'victor':
        service_provider = victor_services.VictorServices()
    else:
        service_provider = gazebo_services.GazeboServices()

    # look up the planner params
    planner, _ = get_planner(planner_params=planner_params,
                             seed=args.seed,
                             verbose=args.verbose)

    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=planner_params['real_time_rate'],
                               max_step_size=planner.fwd_model.max_step_size)

    print(Fore.GREEN + "Running {} Trials".format(args.n_plans) + Fore.RESET)

    runner = EvalPlannerConfigs(
        planner=planner,
        service_provider=service_provider,
        planner_config_name=planner_config_name,
        n_plans_per_env=args.n_plans_per_env,
        n_plans=args.n_plans,
        verbose=args.verbose,
        planner_params=planner_params,
        seed=args.seed,
        outdir=common_output_directory,
        comparison_item_idx=comparison_idx,
        goal=planner_params['fixed_goal'],
        record=args.record,
        pause_between_plans=args.pause_between_plans,
        no_execution=args.no_execution
    )
    runner.run()