import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import ompl.base as ob
import ompl.control as oc
from matplotlib import cm

from moonshine.tests.testing_utils import are_dicts_close_np
import rospy
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.timeout_or_not_progressing import TimeoutOrNotProgressing
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class MyPlannerStatus(Enum):
    Solved = "solved"
    Timeout = "timeout"
    Failure = "failure"
    NotProgressing = "not progressing"

    def __bool__(self):
        if self.value == MyPlannerStatus.Solved:
            return True
        elif self.value == MyPlannerStatus.Timeout:
            return True
        else:
            return False


@dataclass
class PlannerResult:
    path: Optional[List[Dict]]
    actions: Optional[List[Dict]]
    planner_status: MyPlannerStatus


class MyPlanner:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: BaseConstraintChecker,
                 params: Dict,
                 scenario: ExperimentScenario,
                 seed: int,
                 verbose: int,
                 ):
        self.verbose = verbose
        self.planner = None
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.params = params
        # TODO: consider making full env params h/w come from elsewhere. res should match the model though.
        self.si = ob.SpaceInformation(ob.StateSpace())
        self.seed = seed
        self.classifier_rng = np.random.RandomState(seed)
        self.state_sampler_rng = np.random.RandomState(seed)
        self.control_sampler_rng = np.random.RandomState(seed)
        self.scenario = scenario
        self.n_total_action = None
        self.goal_region = None
        self.action_params = self.fwd_model.data_collection_params

        self.state_space = self.scenario.make_ompl_state_space(planner_params=self.params,
                                                               state_sampler_rng=self.state_sampler_rng,
                                                               plot=self.verbose >= 2)
        self.control_space = self.scenario.make_ompl_control_space(self.state_space,
                                                                   self.control_sampler_rng,
                                                                   action_params=self.action_params)

        self.ss = oc.SimpleSetup(self.control_space)

        self.si = self.ss.getSpaceInformation()

        self.ss.setStatePropagator(oc.AdvancedStatePropagatorFn(self.propagate))
        self.ss.setMotionsValidityChecker(oc.MotionsValidityCheckerFn(self.motions_valid))
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))

        # a Dictionary containing the parts of state which are not predicted/planned for, i.e. the environment
        self.environment = None
        self.start_state = None
        self.max_distance_from_start = 0

    def is_valid(self, state):
        return self.state_space.satisfiesBounds(state)

    def motions_valid(self, motions):
        final_state = self.scenario.ompl_state_to_numpy(motions[-1].getState())

        motions_valid = final_state['num_diverged'] < self.classifier_model.horizon - 1  # yes, minus 1
        motions_valid = bool(np.squeeze(motions_valid))
        if not motions_valid:
            self.scenario.plot_rejected_state(final_state)

        # Do some bookkeeping to figure out how the planner is progressing
        distance_from_start = self.scenario.distance(final_state, self.start_state)
        self.max_distance_from_start = max(self.max_distance_from_start, distance_from_start)

        return motions_valid

    def motions_to_numpy(self, motions):
        states_sequence = []
        actions = []
        for t, motion in enumerate(motions):
            # motions is a vector of oc.Motion, which has a state, parent, and control
            state = motion.getState()
            control = motion.getControl()
            state_t = self.scenario.ompl_state_to_numpy(state)
            states_sequence.append(state_t)
            if t > 0:  # skip the first (null) action, because that would represent the action that brings us to the first state
                actions.append(self.scenario.ompl_control_to_numpy(state, control))
        actions = np.array(actions)
        return states_sequence, actions

    def predict(self, previous_states, previous_actions, new_action):
        new_actions = [new_action]
        last_previous_state = previous_states[-1]
        predicted_states = self.fwd_model.propagate(environment=self.environment,
                                                    start_states=last_previous_state,
                                                    actions=new_actions)
        # get only the final state predicted
        final_predicted_state = predicted_states[-1]

        # compute new num_diverged by checking the constraint
        # walk back up the branch until num_diverged == 0
        all_states = [final_predicted_state]
        all_actions = [new_action]
        for previous_idx in range(len(previous_states) - 1, -1, -1):
            previous_state = previous_states[previous_idx]
            all_states.insert(0, previous_state)
            if previous_state['num_diverged'] == 0:
                break
            # this goes after the break because action_i brings you TO state_i and we don't want that last action
            previous_action = previous_actions[previous_idx - 1]
            all_actions.insert(0, previous_action)
        classifier_probabilities = self.classifier_model.check_constraint(environment=self.environment,
                                                                          states_sequence=all_states,
                                                                          actions=all_actions)
        final_classifier_probability = classifier_probabilities[-1]
        if self.verbose >= 2:
            self.scenario.plot_accept_probability(final_classifier_probability)
        if final_classifier_probability > self.params['accept_threshold']:
            final_predicted_state['num_diverged'] = np.array([0.0])
        else:
            final_predicted_state['num_diverged'] = last_previous_state['num_diverged'] + 1
        return final_predicted_state, final_classifier_probability

    def propagate(self, motions, control, duration, state_out):
        del duration  # unused, multi-step propagation is handled inside propagateMotionsWhileValid

        # Convert from OMPL -> Numpy
        previous_states, previous_actions = self.motions_to_numpy(motions)
        previous_state = previous_states[-1]
        previous_ompl_state = motions[-1].getState()
        new_action = self.scenario.ompl_control_to_numpy(previous_ompl_state, control)
        np_final_state, final_classifier_probability = self.predict(previous_states, previous_actions, new_action)

        # Convert back Numpy -> OMPL
        self.scenario.numpy_to_ompl_state(np_final_state, state_out)

        if self.verbose >= 2:
            alpha = final_classifier_probability
            if len(previous_actions) == 0:
                random_color = cm.Dark2(self.control_sampler_rng.uniform(0, 1))
                MyPlanner.propagate.r = random_color[0]
                MyPlanner.propagate.g = random_color[1]
                MyPlanner.propagate.b = random_color[2]
            elif not are_dicts_close_np(previous_actions[-1], new_action):
                random_color = cm.Dark2(self.control_sampler_rng.uniform(0, 1))
                MyPlanner.propagate.r = random_color[0]
                MyPlanner.propagate.g = random_color[1]
                MyPlanner.propagate.b = random_color[2]
                self.scenario.plot_tree_state(previous_state)
                self.scenario.plot_tree_state(MyPlanner.propagate.cached_previous_state)

            self.scenario.plot_current_tree_state(np_final_state, horizon=self.classifier_model.horizon)

            self.scenario.plot_tree_action(previous_state,
                                           new_action,
                                           r=MyPlanner.propagate.r,
                                           g=MyPlanner.propagate.g,
                                           b=MyPlanner.propagate.b,
                                           a=alpha)
            MyPlanner.propagate.cached_previous_state = np_final_state
    propagate.r = 0
    propagate.g = 0
    propagate.b = 0

    def plan(self,
             start_state: Dict,
             environment: Dict,
             goal: Dict,
             ) -> PlannerResult:
        """
        :param start_states: each element is a vector
        :type environment: each element is a vector of state which we don't predict
        :param goal:
        :return: controls, states
        """
        self.environment = environment
        self.goal_region = self.scenario.make_goal_region(self.si,
                                                          rng=self.state_sampler_rng,
                                                          params=self.params,
                                                          goal=goal,
                                                          plot=self.verbose >= 2)

        # create start and goal states
        start_state['stdev'] = np.array([0.0])
        start_state['num_diverged'] = np.array([0.0])
        self.start_state = start_state
        ompl_start_scoped = ob.State(self.state_space)
        self.scenario.numpy_to_ompl_state(start_state, ompl_start_scoped())

        # visualization
        self.scenario.reset_planning_viz()
        self.scenario.plot_environment_rviz(environment)
        self.scenario.plot_start_state(start_state)
        self.scenario.plot_goal(goal, self.params['goal_threshold'])

        self.ss.clear()
        self.ss.setStartState(ompl_start_scoped)
        self.ss.setGoal(self.goal_region)

        ptc = TimeoutOrNotProgressing(self, self.params['termination_criteria'], self.verbose)
        ob_planner_status = self.ss.solve(ptc)
        planner_status = interpret_planner_status(ob_planner_status, ptc)

        if planner_status == MyPlannerStatus.Solved:
            ompl_path = self.ss.getSolutionPath()
            actions, planned_path = self.convert_path(ompl_path)
        else:
            actions = []
            planned_path = [start_state]

        self.goal_region = None
        return PlannerResult(planner_status=planner_status, path=planned_path, actions=actions)

    def convert_path(self, ompl_path: oc.PathControl) -> Tuple[List[Dict], List[Dict]]:
        planned_path = []
        actions = []
        n_actions = ompl_path.getControlCount()
        for time_idx, state in enumerate(ompl_path.getStates()):
            np_state = self.scenario.ompl_state_to_numpy(state)
            planned_path.append(np_state)
            if time_idx < n_actions:
                action = ompl_path.getControl(time_idx)
                action_np = self.scenario.ompl_control_to_numpy(state, action)
                actions.append(action_np)

        return actions, planned_path


def interpret_planner_status(planner_status: ob.PlannerStatus, ptc: TimeoutOrNotProgressing):
    if str(planner_status) == "Exact solution":
        return MyPlannerStatus.Solved
    elif ptc.not_progressing:
        return MyPlannerStatus.NotProgressing
    else:
        return MyPlannerStatus.Timeout
