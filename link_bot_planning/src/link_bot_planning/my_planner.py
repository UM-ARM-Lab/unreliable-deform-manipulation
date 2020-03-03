from dataclasses import dataclass
from dataclasses import dataclass
from typing import Dict

import numpy as np
import ompl.base as ob
import ompl.control as oc
from colorama import Fore

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning.link_bot_goal import LinkBotCompoundGoal
from link_bot_planning.trajectory_smoother import TrajectorySmoother
from link_bot_planning.state_spaces import to_numpy, from_numpy, ValidRopeConfigurationCompoundSampler, \
    TrainingSetCompoundSampler, to_numpy_flat
from link_bot_planning.viz_object import VizObject
from link_bot_pycommon import link_bot_sdf_utils, ros_pycommon
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def ompl_control_to_model_action(control, n_action):
    distance_angle = to_numpy(control, n_action)
    angle = distance_angle[0, 0]
    distance = distance_angle[0, 1]
    np_u = np.array([np.cos(angle) * distance, np.sin(angle) * distance])
    return np_u


@dataclass
class PlannerResult:
    path: Dict[str, np.ndarray]
    controls: np.ndarray
    planner_status: ob.PlannerStatus


class MyPlanner:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: BaseConstraintChecker,
                 params: Dict,
                 services: GazeboServices,
                 viz_object: VizObject,
                 seed: int,
                 ):
        self.planner = None
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.n_action = self.fwd_model.n_action
        self.params = params
        # TODO: consider making full env params h/w come from elsewhere. res should match the model though.
        self.full_env_params = self.fwd_model.full_env_params
        self.services = services
        self.viz_object = viz_object
        self.si = ob.SpaceInformation(ob.StateSpace())
        self.rope_length = fwd_model.hparams['dynamics_dataset_hparams']['rope_length']
        self.seed = seed
        self.classifier_rng = np.random.RandomState(seed)
        self.state_sampler_rng = np.random.RandomState(seed)

        self.state_space = ob.CompoundStateSpace()
        self.state_space_weights = self.params['state_space_weights']

        assert (self.fwd_model.state_keys == list(self.state_space_weights.keys()))

        self.state_space_description = {}
        for subspace_idx, (state_key, weight) in enumerate(self.state_space_weights.items()):
            n_state = self.fwd_model.states_description[state_key]
            subspace_description = {
                "idx": subspace_idx,
                "weight": weight,
                "n_state": n_state
            }
            self.state_space_description[state_key] = subspace_description

            self.link_bot_space = ob.RealVectorStateSpace(n_state)
            self.link_bot_space_bounds = ob.RealVectorBounds(n_state)
            for i in range(n_state):
                if i % 2 == 0:
                    self.link_bot_space_bounds.setLow(i, -self.params['w'] / 2)
                    self.link_bot_space_bounds.setHigh(i, self.params['w'] / 2)
                else:
                    self.link_bot_space_bounds.setLow(i, -self.params['h'] / 2)
                    self.link_bot_space_bounds.setHigh(i, self.params['h'] / 2)
            self.link_bot_space.setBounds(self.link_bot_space_bounds)
            self.state_space.addSubspace(self.link_bot_space, weight=weight)

        self.state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(self.state_sampler_allocator))

        control_bounds = ob.RealVectorBounds(2)
        control_bounds.setLow(0, -np.pi)
        control_bounds.setHigh(0, np.pi)
        control_bounds.setLow(1, 0)
        max_delta_pos = ros_pycommon.get_max_speed() * self.fwd_model.dt
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

        self.goal_description = self.params['goal_description']
        self.goal_subspace_name = self.goal_description['state_key']
        self.goal_subspace_idx = self.state_space_description[self.goal_subspace_name]['idx']
        self.goal_n_state = self.state_space_description[self.goal_subspace_name]['n_state']
        self.goal_point_idx = self.goal_description['point_idx']

        smoothing_params = self.params['smoothing']
        # FIXME actions or controls?
        self.smoother = TrajectorySmoother(fwd_model=fwd_model,
                                           classifier_model=classifier_model,
                                           params=smoothing_params,
                                           goal_point_idx=self.goal_point_idx,
                                           goal_subspace_name=self.goal_subspace_name)

    def is_valid(self, state):
        return self.state_space.satisfiesBounds(state)

    def motions_valid(self, motions):
        # Dict of state_key: np.ndarray [n_state]
        named_states = dict((subspace_name, []) for subspace_name in self.state_space_description.keys())
        actions = []
        for t, motion in enumerate(motions):
            # motions is a vector of oc.Motion, which has a state, parent, and control
            state = motion.getState()
            np_states = self.compound_to_numpy(state)
            for subspace_name, state_t in np_states.items():
                named_states[subspace_name].append(state_t)
            if t > 0:  # skip the first (null) action, because that would represent the action that brings us to the first state
                actions.append(self.control_to_numpy(motion.getControl()))

        named_states_np = {}
        for subspace_name, states in named_states.items():
            named_states_np[subspace_name] = np.array(states)
        actions = np.array(actions)

        accept_probability = self.classifier_model.check_constraint(full_env=self.full_env_data.data,
                                                                    full_env_origin=self.full_env_data.origin,
                                                                    res=self.full_env_data.resolution,
                                                                    states_trajs=named_states_np,
                                                                    actions=actions)

        classifier_accept = accept_probability > self.params['accept_threshold']
        random_accept = self.classifier_rng.uniform(0, 1) <= self.params['random_epsilon']
        motions_is_valid = classifier_accept or random_accept

        if not motions_is_valid:
            final_link_bot_state = named_states['link_bot'][-1]
            self.viz_object.rejected_samples.append(final_link_bot_state)

        return motions_is_valid

    def compound_to_numpy(self, state):
        np_states = {}
        for name, subspace_description in self.state_space_description.items():
            idx = subspace_description['idx']
            n_state = subspace_description['n_state']
            np_s = to_numpy_flat(state[idx], n_state)
            np_states[name] = np_s
        return np_states

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
        final_states = {}
        for state_name, pred_next_states in next_states.items():
            final_states[state_name] = pred_next_states[-1]
        return final_states

    def compound_from_numpy(self, np_final_states: Dict[str, np.ndarray], state_out):
        for name, subspace_description in self.state_space_description.items():
            idx = subspace_description['idx']
            n_state = subspace_description['n_state']
            from_numpy(np_final_states[name], state_out[idx], n_state)

    def propagate(self, start, control, duration, state_out):
        del duration  # unused, multi-step propagation is handled inside propagateMotionsWhileValid

        # Convert from OMPL -> Numpy
        np_states = self.compound_to_numpy(start)
        np_action = self.control_to_numpy(control)
        np_actions = np.expand_dims(np_action, axis=0)

        np_final_states = self.predict(np_states, np_actions)

        # Convert back Numpy -> OMPL
        self.compound_from_numpy(np_final_states, state_out)

    def smooth_path(self,
                    goal_point: np.ndarray,
                    controls: np.ndarray,
                    planned_path: Dict[str, np.ndarray]):
        return self.smoother.smooth(full_env=self.full_env_data.data,
                                    full_env_origin=self.full_env_data.origin,
                                    res=self.full_env_data.resolution,
                                    goal_point=goal_point,
                                    actions=controls,
                                    planned_path=planned_path)

    def plan(self,
             start_states: Dict[str, np.ndarray],
             goal_point: np.ndarray,
             full_env_data: link_bot_sdf_utils.OccupancyData) -> PlannerResult:
        """
        :param full_env_data:
        :param start_states: each element is a vector
        :param goal_point:  1 by n matrix
        :return: controls, states
        """
        self.full_env_data = full_env_data

        # create start and goal states
        compound_start = ob.CompoundState(self.state_space)
        self.compound_from_numpy(start_states, compound_start())

        start = ob.State(compound_start)
        goal = LinkBotCompoundGoal(self.si,
                                   self.goal_description['threshold'],
                                   goal_point,
                                   self.viz_object,
                                   self.goal_subspace_idx,
                                   self.goal_point_idx,
                                   self.goal_n_state)

        self.ss.clear()
        self.viz_object.clear()
        self.ss.setStartState(start)
        self.ss.setGoal(goal)

        planner_status = self.ss.solve(self.params['timeout'])

        if planner_status:
            ompl_path = self.ss.getSolutionPath()
            controls_np, planned_path_dict = self.convert_path(ompl_path)
            controls_np, planned_path_dict = self.smooth_path(goal_point, controls_np, planned_path_dict)
            return PlannerResult(planner_status=planner_status,
                                 path=planned_path_dict,
                                 controls=controls_np)
        return PlannerResult(planner_status)

    def state_sampler_allocator(self, state_space):
        if self.params['sampler_type'] == 'simple':
            sampler = ob.RealVectorStateSampler(state_space)
        elif self.params['sampler_type'] == 'random':
            extent = [-self.params['w'] / 2,
                      self.params['w'] / 2,
                      -self.params['h'] / 2,
                      self.params['h'] / 2]
            # FIXME: need to handle arbitrary state space dictionary/description
            raise NotImplementedError()
            # sampler = ValidRopeConfigurationCompoundSampler(state_space,
            #                                                 my_planner=self,
            #                                                 viz_object=self.viz_object,
            #                                                 extent=extent,
            #                                                 n_rope_state=self.n_state,
            #                                                 subspace_name_to_index=self.subspace_name_to_index,
            #                                                 rope_length=self.rope_length,
            #                                                 max_angle_rad=self.params['max_angle_rad'],
            #                                                 rng=self.state_sampler_rng)
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
        planned_path = {}
        for time_idx, compound_state in enumerate(ompl_path.getStates()):
            for name, subspace_description in self.state_space_description.items():
                if name not in planned_path:
                    planned_path[name] = []

                idx = subspace_description['idx']
                n_state = subspace_description['n_state']
                subspace_state = compound_state[idx]
                planned_path[name].append(to_numpy_flat(subspace_state, n_state))

        # now convert lists to arrays
        planned_path_dict = {}
        for k, v in planned_path.items():
            planned_path_dict[k] = np.array(v)

        np_controls = np.ndarray((ompl_path.getControlCount(), self.n_action))
        for time_idx, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
            # duration is always be 1 for control::RRT, not so for control::SST
            np_controls[time_idx] = ompl_control_to_model_action(control, self.n_action).squeeze()

        return np_controls, planned_path_dict


def interpret_planner_status(planner_status: ob.PlannerStatus, verbose: int = 0):
    if verbose >= 1:
        # If the planner failed, print the error
        if not planner_status:
            print(Fore.RED + planner_status.asString() + Fore.RESET)
