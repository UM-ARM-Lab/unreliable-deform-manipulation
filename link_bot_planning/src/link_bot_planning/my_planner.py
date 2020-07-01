import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import ompl.base as ob
import ompl.control as oc

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

        self.state_space = self.scenario.make_ompl_state_space(self.params, self.state_sampler_rng)
        self.control_space = self.scenario.make_ompl_control_space(
            self.state_space, self.params, self.control_sampler_rng)

        self.ss = oc.SimpleSetup(self.control_space)

        self.si = self.ss.getSpaceInformation()

        self.ss.setStatePropagator(oc.AdvancedStatePropagatorFn(self.propagate))
        self.ss.setMotionsValidityChecker(oc.MotionsValidityCheckerFn(self.motions_valid))
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))

        # a Dictionary containing the parts of state which are not predicted/planned for, i.e. the environment
        self.environment = None

        self.min_distance_to_goal = sys.maxsize

    def is_valid(self, state):
        return self.state_space.satisfiesBounds(state)

    def motions_valid(self, motions):
        final_state = self.scenario.ompl_state_to_numpy(motions[-1].getState())

        # motions_valid = final_state['num_diverged'] < self.classifier_model.horizon - 1  # yes, minus 1
        motions_valid = final_state['num_diverged'] < 1
        motions_valid = bool(np.squeeze(motions_valid))
        if not motions_valid:
            self.scenario.plot_rejected_state(final_state)

        # Do some bookkeeping to figure out how the planner is progressing
        distance_to_goal = self.scenario.distance_to_goal(final_state, self.goal_region.goal)
        self.min_distance_to_goal = min(self.min_distance_to_goal, distance_to_goal)

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
        if self.verbose >= 3:
            raise NotImplementedError()

        # compute new num_diverged by checking the constraint
        # walk back up the branch until num_diverged == 0
        all_states = [final_predicted_state]
        all_actions = [new_action]
        for previous_idx in range(len(previous_states) - 1, -1, -1):
            previous_state = previous_states[previous_idx]
            all_states.insert(0, previous_state)
            if self.verbose >= 3:
                raise NotImplementedError()
            if previous_state['num_diverged'] == 0:
                break
            # this goes after the break because action_i brings you TO state_i and we don't want that last action
            previous_action = previous_actions[previous_idx - 1]
            all_actions.insert(0, previous_action)
            if self.verbose >= 3:
                raise NotImplementedError()
        classifier_probabilities = self.classifier_model.check_constraint(environment=self.environment,
                                                                          states_sequence=all_states,
                                                                          actions=all_actions)
        final_classifier_probability = classifier_probabilities[-1]
        classifier_accept = final_classifier_probability > self.params['accept_threshold']
        final_predicted_state['num_diverged'] = np.array(
            [0.0]) if classifier_accept else last_previous_state['num_diverged'] + 1
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
            self.scenario.plot_tree_action(previous_state, new_action, alpha=final_classifier_probability)
            self.scenario.plot_tree_state(np_final_state)

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
        self.min_distance_to_goal = sys.maxsize
        self.goal_region = self.scenario.make_goal_region(
            self.si, rng=self.state_sampler_rng, params=self.params, goal=goal)

        # create start and goal states
        start_state['stdev'] = np.array([0.0])
        start_state['num_diverged'] = np.array([0.0])
        ompl_start_scoped = ob.State(self.state_space)
        self.scenario.numpy_to_ompl_state(start_state, ompl_start_scoped())

        self.scenario.reset_planning_viz()
        if self.verbose >= 2:
            self.scenario.plot_environment_rviz(environment)
            self.scenario.plot_start_state(start_state)

        self.ss.clear()
        self.ss.setStartState(ompl_start_scoped)
        self.ss.setGoal(self.goal_region)

        if self.verbose >= 3:
            raise NotImplementedError()

        ptc = TimeoutOrNotProgressing(self, self.params['termination_criteria'], self.verbose)
        ob_planner_status = self.ss.solve(ptc)
        planner_status = interpret_planner_status(ob_planner_status, ptc)

        if planner_status == planner_status.Failure:
            self.goal_region = None
            return PlannerResult(planner_status=planner_status, path=None, actions=None)
        else:
            ompl_path = self.ss.getSolutionPath()
            actions, planned_path = self.convert_path(ompl_path)
            self.goal_region = None
            return PlannerResult(planner_status=planner_status, path=planned_path, actions=actions)

    def convert_path(self, ompl_path: oc.PathControl) -> Tuple[List[Dict], List[Dict]]:
        planned_path = []
        actions = []
        n_actions = ompl_path.getControlCount()
        for time_idx, state in enumerate(ompl_path.getStates()):
            np_state = self.scenario.ompl_state_to_numpy(state)
            action = ompl_path.getControl(time_idx)
            planned_path.append(np_state)
            if time_idx < n_actions:
                action_np = self.scenario.ompl_control_to_numpy(state, action)
                actions.append(action_np)

        return actions, planned_path


def interpret_planner_status(planner_status: ob.PlannerStatus, ptc: TimeoutOrNotProgressing):
    if planner_status == "Exact solution":
        return MyPlannerStatus.Solved
    elif ptc.not_progressing:
        return MyPlannerStatus.NotProgressing
    elif ptc.timed_out:
        return MyPlannerStatus.Timeout
    else:
        return MyPlannerStatus.Failure
