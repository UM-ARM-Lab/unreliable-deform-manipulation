import pathlib
import tempfile
import uuid
from time import time, sleep
from typing import Optional, Dict, List, Tuple

import hjson
import numpy as np
from colorama import Fore
from ompl import util as ou

import rosbag
import rospy
from arc_utilities.conditional_try import conditional_try
from link_bot_gazebo_python import gazebo_services
from link_bot_planning import plan_and_execute
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.serialization import dummy_proof_write, my_dump, my_hdump
from moonshine.moonshine_utils import numpify


class EvalPlannerConfigs(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 service_provider: BaseServices,
                 trials: List[int],
                 verbose: int,
                 planner_params: Dict,
                 outdir: pathlib.Path,
                 record: Optional[bool] = False,
                 no_execution: Optional[bool] = False,
                 test_scenes_dir: Optional[pathlib.Path] = None,
                 save_test_scenes_dir: Optional[pathlib.Path] = None,
                 ):
        super().__init__(planner,
                         trials=trials,
                         verbose=verbose,
                         planner_params=planner_params,
                         service_provider=service_provider,
                         test_scenes_dir=test_scenes_dir,
                         save_test_scenes_dir=save_test_scenes_dir,
                         no_execution=no_execution)
        self.record = record
        self.outdir = outdir

        self.outdir.mkdir(parents=True, exist_ok=True)
        rospy.loginfo(Fore.BLUE + f"Output directory: {self.outdir.as_posix()}")

        metadata = {
            "trials":         self.trials,
            "planner_params": self.planner_params,
            "scenario":       self.planner.scenario.simple_name(),
        }
        metadata.update(self.planner.get_metadata())
        with (self.outdir / 'metadata.json').open("w") as metadata_file:
            my_dump(metadata, metadata_file, indent=2)

        self.bag = None
        self.final_execution_to_goal_errors = []

    def randomize_environment(self):
        if self.verbose >= 1:
            rospy.loginfo("Randomizing env")
        self.service_provider.play()
        super().randomize_environment()
        self.service_provider.pause()
        if self.verbose >= 1:
            rospy.loginfo("End randomizing env")

    def on_start_trial(self, trial_idx: int):
        if self.record:
            filename = self.outdir.absolute() / 'plan-{}.avi'.format(trial_idx)
            self.service_provider.start_record_trial(str(filename))
            bagname = self.outdir.absolute() / f"follow_joint_trajectory_goal_{trial_idx}.bag"
            rospy.loginfo(Fore.YELLOW + f"Saving bag file name: {bagname.as_posix()}")
            self.bag = rosbag.Bag(bagname, 'w')

    def follow_joint_trajectory_goal_callback(self, goal_msg):
        if self.record:
            self.bag.write('/both_arms_controller/follow_joint_trajectory/goal', goal_msg)
            self.bag.flush()

    def on_trial_complete(self, trial_data: Dict, trial_idx: int):
        extra_trial_data = {
            "planner_params": self.planner_params,
            "scenario":       self.planner.scenario.simple_name(),
            'current_time':   int(time()),
            'uuid':           uuid.uuid4(),
        }
        trial_data.update(extra_trial_data)
        data_filename = self.outdir / f'{trial_idx}_metrics.json.gz'
        dummy_proof_write(trial_data, data_filename)

        if self.record:
            # TODO: maybe make this happen async?
            sleep(1)
            self.service_provider.stop_record_trial()
            self.bag.close()

        # compute the current running average success
        goal = trial_data['planning_queries'][0].goal
        final_actual_state = numpify(trial_data['end_state'])
        final_execution_to_goal_error = self.planner.scenario.distance_to_goal(final_actual_state, goal)
        self.final_execution_to_goal_errors.append(final_execution_to_goal_error)
        goal_threshold = self.planner_params['goal_params']['threshold']
        n = len(self.final_execution_to_goal_errors)
        success_percentage = np.count_nonzero(np.array(self.final_execution_to_goal_errors) < goal_threshold) / n * 100
        update_msg = f"[{self.outdir.stem}] Current average success rate {success_percentage:.2f}%"
        rospy.loginfo(Fore.LIGHTBLUE_EX + update_msg)


def evaluate_planning_method(planner_params: Dict,
                             trials: List[int],
                             comparison_root_dir: pathlib.Path,
                             verbose: int = 0,
                             record: bool = False,
                             no_execution: bool = False,
                             timeout: Optional[int] = None,
                             test_scenes_dir: Optional[pathlib.Path] = None,
                             save_test_scenes_dir: Optional[pathlib.Path] = None,
                             ):
    # override some arguments
    if timeout is not None:
        rospy.loginfo(f"Overriding with timeout {timeout}")
        planner_params["termination_criteria"]['timeout'] = timeout

    # Start Services
    service_provider = gazebo_services.GazeboServices()
    planner = get_planner(planner_params=planner_params, verbose=verbose)

    service_provider.setup_env(verbose=verbose,
                               real_time_rate=planner_params['real_time_rate'],
                               max_step_size=planner.fwd_model.max_step_size,
                               play=False)

    # FIXME: RAII -- you should not be able to call get_state on a scenario until this method has been called
    #  which could be done by making a type, something like "EmbodiedScenario" which has get_state and execute_action,
    planner.scenario.on_before_get_state_or_execute_action()

    runner = EvalPlannerConfigs(
        planner=planner,
        service_provider=service_provider,
        trials=trials,
        verbose=verbose,
        planner_params=planner_params,
        outdir=comparison_root_dir,
        test_scenes_dir=test_scenes_dir,
        save_test_scenes_dir=save_test_scenes_dir,
        record=record,
        no_execution=no_execution
    )
    runner.run()


def read_logfile(logfile_name):
    with logfile_name.open("r") as logfile:
        log = hjson.load(logfile)
    return log


def write_logfile(log, logfile_name):
    with logfile_name.open("w") as logfile:
        my_hdump(log, logfile)
    return log


def planning_evaluation(outdir: pathlib.Path,
                        planners_params: List[Tuple[str, Dict]],
                        trials: List[int],
                        logfile_name: Optional[str],
                        skip_on_exception: Optional[bool] = False,
                        verbose: int = 0,
                        record: bool = False,
                        no_execution: bool = False,
                        timeout: Optional[int] = None,
                        test_scenes_dir: Optional[pathlib.Path] = None,
                        save_test_scenes_dir: Optional[pathlib.Path] = None,
                        ):
    ou.setLogLevel(ou.LOG_ERROR)

    if logfile_name is None:
        logfile_name = pathlib.Path(tempfile.gettempdir()) / f'planning-evaluation-log-file-{time()}'

    log = read_logfile(logfile_name)

    rospy.loginfo(Fore.CYAN + "common output directory: {}".format(outdir))
    if not outdir.is_dir():
        rospy.loginfo(Fore.YELLOW + "Creating output directory: {}".format(outdir))
        outdir.mkdir(parents=True)

    for comparison_idx, (planner_config_name, planner_params) in enumerate(planners_params):
        log[comparison_idx] = outdir

        rospy.loginfo(Fore.GREEN + f"Running method {planner_config_name}")
        subfolder = f"{planner_config_name}_{comparison_idx}"
        comparison_root_dir = outdir / subfolder

        conditional_try(skip_on_exception,
                        evaluate_planning_method,
                        planner_params=planner_params,
                        trials=trials,
                        comparison_root_dir=comparison_root_dir,
                        verbose=verbose,
                        record=record,
                        no_execution=no_execution,
                        timeout=timeout,
                        test_scenes_dir=test_scenes_dir,
                        save_test_scenes_dir=save_test_scenes_dir,
                        )

        rospy.loginfo(f"Results written to {outdir}")
        log[comparison_idx] = outdir

        write_logfile(log, logfile_name)

    return outdir
