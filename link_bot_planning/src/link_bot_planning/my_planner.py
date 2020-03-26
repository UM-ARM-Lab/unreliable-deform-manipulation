from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import ompl.base as ob
import ompl.control as oc
import rospy
from colorama import Fore

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_planning.link_bot_goal import MyGoalRegion
from link_bot_planning.state_spaces import from_numpy, to_numpy_flat, ValidRopeConfigurationCompoundSampler, \
    compound_to_numpy, ompl_control_to_model_action
from link_bot_planning.trajectory_smoother import TrajectorySmoother
from link_bot_planning.viz_object import VizObject
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.ros_pycommon import Services
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


@dataclass
class PlannerResult:
    path: Optional[List[Dict]]
    actions: Optional[np.ndarray]
    planner_status: ob.PlannerStatus


class MyPlanner:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: BaseConstraintChecker,
                 params: Dict,
                 service_provider: Services,
                 scenario: ExperimentScenario,
                 viz_object: VizObject,
                 seed: int,
                 verbose: int,
                 ):
        # FIXME:
        self.verbose = verbose
        self.rope_length = rospy.get_param('/link_bot/rope_length')
        self.planner = None
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.n_action = self.fwd_model.n_action
        self.params = params
        # TODO: consider making full env params h/w come from elsewhere. res should match the model though.
        self.full_env_params = self.fwd_model.full_env_params
        self.service_provider = service_provider
        self.viz_object = viz_object
        self.si = ob.SpaceInformation(ob.StateSpace())
        self.seed = seed
        self.classifier_rng = np.random.RandomState(seed)
        self.state_sampler_rng = np.random.RandomState(seed)
        self.experiment_scenario = scenario

        self.state_space_description = {}
        self.state_space = ob.CompoundStateSpace()
        self.subspaces = []
        self.subspace_bounds = []
        for subspace_idx, state_key in enumerate(self.fwd_model.states_keys):
            weight = self.experiment_scenario.get_subspace_weight(state_key)
            n_state = self.fwd_model.states_description[state_key]
            subspace_description = {
                "idx": subspace_idx,
                "weight": weight,
                "n_state": n_state
            }
            self.state_space_description[state_key] = subspace_description

            subspace = ob.RealVectorStateSpace(n_state)
            bounds = ob.RealVectorBounds(n_state)
            for i in range(n_state):
                if i % 2 == 0:
                    bounds.setLow(i, -self.params['full_env_w'] / 2)
                    bounds.setHigh(i, self.params['full_env_w'] / 2)
                else:
                    bounds.setLow(i, -self.params['full_env_h'] / 2)
                    bounds.setHigh(i, self.params['full_env_h'] / 2)
            subspace.setBounds(bounds)
            self.subspaces.append(subspace_idx)
            self.subspace_bounds.append(bounds)
            self.state_space.addSubspace(subspace, weight=weight)

        self.state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(self.state_sampler_allocator))

        control_bounds = ob.RealVectorBounds(2)
        control_bounds.setLow(0, -np.pi)
        control_bounds.setHigh(0, np.pi)
        control_bounds.setLow(1, 0)
        max_delta_pos = service_provider.get_max_speed() * self.fwd_model.dt
        control_bounds.setHigh(1, max_delta_pos)
        self.control_space = oc.RealVectorControlSpace(self.state_space, self.n_action)
        self.control_space.setBounds(control_bounds)

        self.ss = oc.SimpleSetup(self.control_space)

        self.si = self.ss.getSpaceInformation()

        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))
        self.ss.setMotionsValidityChecker(oc.MotionsValidityCheckerFn(self.motions_valid))
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))

        if params['directed_control_sampler'] == 'simple':
            pass  # the default
        else:
            raise NotImplementedError()

        self.full_env_data = None

        if params['sampler_type'] == 'sample_train':
            raise NotImplementedError()

        smoothing_params = self.params['smoothing']
        # FIXME: call it "action" instead of control everywhere
        if smoothing_params is None:
            self.smoother = None
        else:
            self.smoother = TrajectorySmoother(verbose=self.verbose,
                                               fwd_model=fwd_model,
                                               classifier_model=classifier_model,
                                               params=smoothing_params,
                                               experiment_scenario=self.experiment_scenario)

    def is_valid(self, state):
        return self.state_space.satisfiesBounds(state)

    def motions_valid(self, motions):
        states_sequence = []
        actions = []
        for t, motion in enumerate(motions):
            # motions is a vector of oc.Motion, which has a state, parent, and control
            state = motion.getState()
            state_t = compound_to_numpy(self.state_space_description, state)
            states_sequence.append(state_t)
            if t > 0:  # skip the first (null) action, because that would represent the action that brings us to the first state
                actions.append(self.control_to_numpy(motion.getControl()))

        actions = np.array(actions)

        accept_probability = self.classifier_model.check_constraint(full_env=self.full_env_data.data,
                                                                    full_env_origin=self.full_env_data.origin,
                                                                    res=self.full_env_data.resolution,
                                                                    states_sequence=states_sequence,
                                                                    actions=actions)

        classifier_accept = accept_probability > self.params['accept_threshold']
        random_accept = self.classifier_rng.uniform(0, 1) <= self.params['random_epsilon']
        motions_is_valid = classifier_accept or random_accept

        if random_accept and not classifier_accept:
            final_link_bot_state = states_sequence[-1]
            self.viz_object.randomly_accepted_samples.append(final_link_bot_state)

        if not motions_is_valid:
            final_link_bot_state = states_sequence[-1]
            self.viz_object.rejected_samples.append(final_link_bot_state)

        return motions_is_valid

    def control_to_numpy(self, control):
        np_u = ompl_control_to_model_action(control, self.n_action)
        return np_u

    def predict(self, np_states, np_actions):
        # use the forward model to predict the next configuration
        # NOTE full env here could be different than the full env the classifier gets? Maybe classifier should sub-select from
        #  the actual full env?
        next_states = self.fwd_model.propagate(full_env=self.full_env_data.data,
                                               full_env_origin=self.full_env_data.origin,
                                               res=self.fwd_model.full_env_params.res,
                                               start_states=np_states,
                                               actions=np_actions)
        # get only the final state predicted
        final_states = next_states[-1]
        return final_states

    def compound_from_numpy(self, np_final_states: Dict[str, np.ndarray], state_out):
        for subspace_name, subspace_description in self.state_space_description.items():
            idx = subspace_description['idx']
            n_state = subspace_description['n_state']
            from_numpy(np_final_states[subspace_name], state_out[idx], n_state)

    def propagate(self, start, control, duration, state_out):
        del duration  # unused, multi-step propagation is handled inside propagateMotionsWhileValid

        # Convert from OMPL -> Numpy
        np_states = compound_to_numpy(self.state_space_description, start)
        np_action = self.control_to_numpy(control)
        np_actions = np.expand_dims(np_action, axis=0)

        np_final_states = self.predict(np_states, np_actions)

        # Convert back Numpy -> OMPL
        self.compound_from_numpy(np_final_states, state_out)

    def smooth_path(self,
                    goal,
                    controls: np.ndarray,
                    planned_path: List[Dict]):
        smoothed_actions_tf, smoothed_path_tf = self.smoother.smooth(full_env=self.full_env_data.data,
                                                                     full_env_origin=self.full_env_data.origin,
                                                                     res=self.full_env_data.resolution,
                                                                     goal=goal,
                                                                     actions=controls,
                                                                     planned_path=planned_path)
        smoothed_path = []
        for state_tf in smoothed_path_tf:
            state_np = {}
            for k, v in state_tf.items():
                state_np[k] = v.numpy()
            smoothed_path.append(state_np)

        return smoothed_actions_tf.numpy(), smoothed_path

    def plan(self,
             start_states: Dict[str, np.ndarray],
             goal,
             full_env_data: link_bot_sdf_utils.OccupancyData) -> PlannerResult:
        """
        :param full_env_data:
        :param start_states: each element is a vector
        :param goal:
        :return: controls, states
        """
        self.full_env_data = full_env_data

        # create start and goal states
        ompl_start = ob.CompoundState(self.state_space)
        self.compound_from_numpy(start_states, ompl_start())

        start = ob.State(ompl_start)
        ompl_goal = MyGoalRegion(self.si,
                                 self.params['goal_threshold'],
                                 goal,
                                 self.viz_object,
                                 self.experiment_scenario,
                                 self.state_space_description)

        self.ss.clear()
        self.viz_object.clear()
        self.ss.setStartState(start)
        self.ss.setGoal(ompl_goal)

        planner_status = self.ss.solve(self.params['timeout'])

        if planner_status:
            ompl_path = self.ss.getSolutionPath()
            controls_np, planned_path = self.convert_path(ompl_path)
            if self.smoother is not None:
                controls_np, planned_path = self.smooth_path(goal, controls_np, planned_path)
            return PlannerResult(planner_status=planner_status,
                                 path=planned_path,
                                 actions=controls_np)
        return PlannerResult(planner_status=planner_status,
                             path=None,
                             actions=None)

    def state_sampler_allocator(self, state_space):
        # Note: I had issues using RealVectorStateSampler() here...
        if self.params['sampler_type'] == 'random':
            extent = [-self.params['w'] / 2,
                      self.params['w'] / 2,
                      -self.params['h'] / 2,
                      self.params['h'] / 2]
            # FIXME: need to handle arbitrary state space dictionary/description
            #  this is such a hack
            sampler = ValidRopeConfigurationCompoundSampler(state_space,
                                                            my_planner=self,
                                                            viz_object=self.viz_object,
                                                            extent=extent,
                                                            link_bot_state_idx=self.state_space_description['link_bot']['idx'],
                                                            n_rope_state=self.state_space_description['link_bot']['n_state'],
                                                            rope_length=self.rope_length,
                                                            max_angle_rad=self.params['max_angle_rad'],
                                                            rng=self.state_sampler_rng)
        elif self.params['sampler_type'] == 'sample_train':
            raise NotImplementedError()
            # sampler = TrainingSetCompoundSampler(state_space,
            #                                      self.viz_object,
            #                                      train_dataset=self.training_dataset,
            #                                      sequence_length=self.train_dataset_max_sequence_length,
            #                                      rng=self.state_sampler_rng)
        else:
            raise ValueError("Invalid sampler type {}".format(self.params['sampler_type']))

        return sampler

    def convert_path(self, ompl_path: oc.PathControl):
        planned_path = []
        for time_idx, compound_state in enumerate(ompl_path.getStates()):
            state = {}
            for name, subspace_description in self.state_space_description.items():
                idx = subspace_description['idx']
                n_state = subspace_description['n_state']
                subspace_state = compound_state[idx]
                state[name] = to_numpy_flat(subspace_state, n_state)
            planned_path.append(state)

        np_controls = np.ndarray((ompl_path.getControlCount(), self.n_action))
        for time_idx, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
            # duration is always be 1 for control::RRT, not so for control::SST
            np_controls[time_idx] = ompl_control_to_model_action(control, self.n_action).squeeze()

        return np_controls, planned_path


def interpret_planner_status(planner_status: ob.PlannerStatus, verbose: int = 0):
    if verbose >= 1:
        # If the planner failed, print the error
        if not planner_status:
            print(Fore.RED + planner_status.asString() + Fore.RESET)
