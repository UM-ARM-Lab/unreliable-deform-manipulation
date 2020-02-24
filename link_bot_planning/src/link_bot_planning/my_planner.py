import pathlib
from typing import Dict

import numpy as np
import ompl.base as ob
import ompl.control as oc
from colorama import Fore

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning.link_bot_goal import LinkBotCompoundGoal
from link_bot_planning.random_directed_control_sampler import RandomDirectedControlSampler
from link_bot_planning.state_spaces import to_numpy, from_numpy, ValidRopeConfigurationCompoundSampler, \
    TrainingSetCompoundSampler, to_numpy_local_env, to_numpy_flat
from link_bot_planning.viz_object import VizObject
from link_bot_pycommon import link_bot_sdf_utils, link_bot_pycommon, ros_pycommon
from link_bot_pycommon.ros_pycommon import get_local_occupancy_data
from state_space_dynamics.base_forward_model import BaseForwardModel


def ompl_control_to_model_action(control, n_control):
    distance_angle = to_numpy(control, n_control)
    angle = distance_angle[0, 0]
    distance = distance_angle[0, 1]
    # action here needs to be batch_size,sequence_length,n_control == 1,1,2
    np_u = np.array([[[np.cos(angle) * distance, np.sin(angle) * distance]]])
    return np_u


class MyPlanner:

    def __init__(self,
                 fwd_model: BaseForwardModel,
                 classifier_model: BaseClassifier,
                 planner_params: Dict,
                 services: GazeboServices,
                 viz_object: VizObject,
                 seed: int,
                 ):
        self.planner = None
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        # TODO: remove this concept, it's ill posed. we need Dict[str, int]
        self.n_state = self.fwd_model.n_state
        self.n_links = link_bot_pycommon.n_state_to_n_links(self.n_state)
        self.n_control = self.fwd_model.n_control
        self.n_tether_state = ros_pycommon.get_n_tether_state()
        self.full_env_params = self.fwd_model.full_env_params
        self.planner_params = planner_params
        self.services = services
        self.viz_object = viz_object
        self.si = ob.SpaceInformation(ob.StateSpace())
        self.rope_length = fwd_model.hparams['dynamics_dataset_hparams']['rope_length']
        self.seed = seed
        self.classifier_rng = np.random.RandomState(seed)
        self.state_sampler_rng = np.random.RandomState(seed)

        self.state_space = ob.CompoundStateSpace()
        self.state_space_description = self.planner_params['state_space']

        self.subspace_name_to_index = {}
        self.subspaces_to_plan_with = {}
        for subspace_idx, (name, component_description) in enumerate(self.state_space_description.items()):
            self.subspace_name_to_index[name] = subspace_idx
            if name == 'local_env':
                self.n_local_env = self.fwd_model.local_env_params.w_cols * self.fwd_model.local_env_params.h_rows
                self.local_env_space = ob.RealVectorStateSpace(self.n_local_env)
                epsilon = 1e-3
                self.local_env_space.setBounds(-epsilon, 1 + epsilon)
                self.state_space.addSubspace(self.local_env_space, weight=component_description['weight'])
            elif name == 'local_env_origin':
                self.local_env_origin_space = ob.RealVectorStateSpace(2)
                self.local_env_origin_space.setBounds(-10000, 10000)
                self.state_space.addSubspace(self.local_env_origin_space, weight=component_description['weight'])
            elif name == 'link_bot':
                self.link_bot_space = ob.RealVectorStateSpace(self.n_state)
                self.link_bot_space_bounds = ob.RealVectorBounds(self.n_state)
                self.subspaces_to_plan_with[name] = (subspace_idx, self.n_state)
                for i in range(self.n_state):
                    if i % 2 == 0:
                        self.link_bot_space_bounds.setLow(i, -self.planner_params['w'] / 2)
                        self.link_bot_space_bounds.setHigh(i, self.planner_params['w'] / 2)
                    else:
                        self.link_bot_space_bounds.setLow(i, -self.planner_params['h'] / 2)
                        self.link_bot_space_bounds.setHigh(i, self.planner_params['h'] / 2)
                self.link_bot_space.setBounds(self.link_bot_space_bounds)
                self.state_space.addSubspace(self.link_bot_space, weight=component_description['weight'])
            elif name == 'tether':
                self.tether_space = ob.RealVectorStateSpace(self.n_tether_state)
                self.tether_space_bounds = ob.RealVectorBounds(self.n_tether_state)
                self.subspaces_to_plan_with[name] = (subspace_idx, self.n_tether_state)
                for i in range(self.n_tether_state):
                    if i % 2 == 0:
                        self.tether_space_bounds.setLow(i, -self.planner_params['w'] / 2)
                        self.tether_space_bounds.setHigh(i, self.planner_params['w'] / 2)
                    else:
                        self.tether_space_bounds.setLow(i, -self.planner_params['h'] / 2)
                        self.tether_space_bounds.setHigh(i, self.planner_params['h'] / 2)
                self.tether_space.setBounds(self.tether_space_bounds)
                self.state_space.addSubspace(self.tether_space, weight=component_description['weight'])
        self.local_env_subspace_idx = self.subspace_name_to_index['local_env']
        self.local_env_origin_subspace_idx = self.subspace_name_to_index['local_env_origin']

        self.state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(self.state_sampler_allocator))

        control_bounds = ob.RealVectorBounds(2)
        control_bounds.setLow(0, -np.pi)
        control_bounds.setHigh(0, np.pi)
        control_bounds.setLow(1, 0)
        max_delta_pos = ros_pycommon.get_max_speed() * self.fwd_model.dt
        control_bounds.setHigh(1, max_delta_pos)
        self.control_space = oc.RealVectorControlSpace(self.state_space, self.n_control)
        self.control_space.setBounds(control_bounds)

        self.ss = oc.SimpleSetup(self.control_space)

        self.si = self.ss.getSpaceInformation()

        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))

        if planner_params['directed_control_sampler'] == 'simple':
            pass  # the default
        elif planner_params['directed_control_sampler'] == 'random':
            raise ValueError("This DCS breaks nearest neighbor somehow")
            self.si.setDirectedControlSamplerAllocator(RandomDirectedControlSampler.allocator(self.seed, self))

        self.full_env = None
        self.full_env_origin = None

        if planner_params['sampler_type'] == 'sample_train':
            self.dataset_dirs = [pathlib.Path(p) for p in self.fwd_model.hparams['datasets']]
            dataset = LinkBotStateSpaceDataset(self.dataset_dirs)
            tf_dataset = dataset.get_datasets(mode='train', sequence_length=None)
            tf_dataset = tf_dataset.shuffle(seed=args.seed, shuffle=True)
            self.training_dataset = tf_dataset
            self.train_dataset_max_sequence_length = dataset.max_sequence_length

    def get_local_env_at(self, x: float, y: float):
        center_point = np.array([x, y])
        return get_local_occupancy_data(rows=self.fwd_model.local_env_params.h_rows,
                                        cols=self.fwd_model.local_env_params.w_cols,
                                        res=self.fwd_model.local_env_params.res,
                                        center_point=center_point,
                                        services=self.services)

    def is_valid(self, state):
        for idx, _ in self.subspaces_to_plan_with.values():
            return self.state_space.getSubspace(idx).satisfiesBounds(state[idx])

    def edge_is_valid(self, states: Dict[str, np.ndarray], np_u: np.ndarray, states_next: Dict[str, np.ndarray]):
        np_s = states['link_bot']
        np_s_next = states_next['link_bot']

        np_s = np_s.flatten()
        np_s_next = np_s_next.flatten()
        np_u = np_u.flatten()
        local_env_data = self.get_local_env_at(np_s[-2], np_s[-1])

        accept_probability = self.classifier_model.predict(local_env_data=local_env_data, s1=np_s, s2=np_s_next, action=np_u)
        p = self.classifier_rng.uniform(0, 1)
        # classifier_accept = p <= accept_probability
        classifier_accept = accept_probability > self.planner_params['accept_threshold']  # FIXME: use the probability
        # FIXME: put random epsilon back in
        # FIXME: compute random_epsilon as e^(-k*validation_accuracy_of_classifier)
        # for example, my validation accuracy is ~0.85, so if k=3.4 I get 0.05
        random_accept = self.classifier_rng.uniform(0, 1) <= self.planner_params['random_epsilon']
        # edge_is_valid = classifier_accept or random_accept
        edge_is_valid = classifier_accept  # or random_accept
        return edge_is_valid

    def propagate(self, start, control, duration, state_out):
        del duration  # unused, multi-step propagation is handled inside propagateWhileValid
        states = {}
        for name, (idx, n) in self.subspaces_to_plan_with.items():
            np_s = to_numpy_flat(start[idx], n)
            states[name] = np_s

        np_u = ompl_control_to_model_action(control, self.n_control)

        # use the forward model to predict the next configuration
        next_states = self.fwd_model.predict(full_env=self.full_env,
                                             full_env_origin=self.full_env_origin,
                                             res=self.fwd_model.full_env_params.res,
                                             states=states,
                                             actions=np_u[0])

        # validate the edge
        # TODO: implement this inside OMPL
        edge_is_valid = self.edge_is_valid(states, np_u, next_states)

        # copy the result into the ompl state data structure
        if not edge_is_valid:
            # This will ensure this edge is not added to the tree
            link_bot_subspace_idx = self.subspaces_to_plan_with['link_bot'][0]
            state_out[link_bot_subspace_idx][0] = 1000
            self.viz_object.rejected_samples.append(next_states['link_bot'])
        else:
            for name, (idx, n) in self.subspaces_to_plan_with.items():
                # -1 gets the last prediction, but since all actions are of length 1, this is also the second element
                next_state = next_states[name][-1]
                from_numpy(next_state, state_out[idx], n)

            head_point = states['link_bot'][-2:]
            local_env_data_next = self.get_local_env_at(head_point[0], head_point[1])
            local_env = local_env_data_next.data.flatten().astype(np.float64)
            for idx in range(self.n_local_env):
                occupancy_value = local_env[idx]
                state_out[self.local_env_subspace_idx][idx] = occupancy_value
            origin_next = local_env_data_next.origin.astype(np.float64)
            # FIXME: 2 is hardcoded
            from_numpy(origin_next, state_out[self.local_env_origin_subspace_idx], 2)

    # TODO: make this return a data structure. something with a name
    def plan(self, start_states: Dict[str, np.ndarray], tail_goal_point: np.ndarray, full_env_data: link_bot_sdf_utils.OccupancyData):
        """
        :param full_env_data:
        :param start_states: each element is a vector
        :param tail_goal_point:  1 by n matrix
        :return: controls, states
        """
        self.full_env = full_env_data.data
        self.full_env_origin = full_env_data.origin

        # create start and goal states
        link_bot_start_state = start_states['link_bot']
        start_local_occupancy = self.get_local_env_at(link_bot_start_state[-2], link_bot_start_state[-1])
        compound_start = ob.CompoundState(self.state_space)
        for name, (idx, n) in self.subspaces_to_plan_with.items():
            for i in range(n):
                compound_start()[idx][i] = start_states[name][i]
        start_local_occupancy_flat_double = start_local_occupancy.data.flatten().astype(np.float64)
        for idx in range(self.n_local_env):
            occupancy_value = start_local_occupancy_flat_double[idx]
            compound_start()[self.local_env_subspace_idx][idx] = occupancy_value
        start_local_occupancy_origin_double = start_local_occupancy.origin.astype(np.float64)
        compound_start()[self.local_env_origin_subspace_idx][0] = start_local_occupancy_origin_double[0]
        compound_start()[self.local_env_origin_subspace_idx][1] = start_local_occupancy_origin_double[1]

        start = ob.State(compound_start)
        goal = LinkBotCompoundGoal(self.si, self.planner_params['goal_threshold'], tail_goal_point, self.viz_object, self.n_state)

        self.ss.clear()
        self.viz_object.clear()
        self.ss.setStartState(start)
        self.ss.setGoal(goal)

        solved = self.ss.solve(self.planner_params['timeout'])

        if solved:
            ompl_path = self.ss.getSolutionPath()
            return self.convert_path(ompl_path, full_env_data, solved)
        return None, None, full_env_data, solved

    def state_sampler_allocator(self, state_space):
        if self.planner_params['sampler_type'] == 'random':
            extent = [-self.planner_params['w'] / 2,
                      self.planner_params['w'] / 2,
                      -self.planner_params['h'] / 2,
                      self.planner_params['h'] / 2]
            # FIXME: need to handle arbitrary state space dictionary/description
            sampler = ValidRopeConfigurationCompoundSampler(state_space,
                                                            my_planner=self,
                                                            viz_object=self.viz_object,
                                                            extent=extent,
                                                            n_rope_state=self.n_state,
                                                            subspace_name_to_index=self.subspace_name_to_index,
                                                            rope_length=self.rope_length,
                                                            max_angle_rad=self.planner_params['max_angle_rad'],
                                                            rng=self.state_sampler_rng)
        elif self.planner_params['sampler_type'] == 'sample_train':
            sampler = TrainingSetCompoundSampler(state_space,
                                                 self.viz_object,
                                                 train_dataset=self.training_dataset,
                                                 sequence_length=self.train_dataset_max_sequence_length,
                                                 rng=self.state_sampler_rng)
        else:
            raise ValueError("Invalid sampler type {}".format(self.planner_params['sampler_type']))

        return sampler

    def convert_path(self, ompl_path: ob.Path, full_env_data: link_bot_sdf_utils.OccupancyData, solved: bool):
        planned_path = {}
        for time_idx, compound_state in enumerate(ompl_path.getStates()):
            for subspace_idx, name in enumerate(self.state_space_description):
                if name not in planned_path:
                    planned_path[name] = []

                subspace_state = compound_state[subspace_idx]
                if name == 'local_env':
                    planned_path[name].append(to_numpy_local_env(subspace_state,
                                                                 self.fwd_model.local_env_params.w_cols,
                                                                 self.fwd_model.local_env_params.h_rows))
                elif name == 'local_env_origin':
                    planned_path[name].append(to_numpy_flat(subspace_state, 2))
                elif name == 'link_bot':
                    planned_path[name].append(to_numpy_flat(subspace_state, self.n_state))
                elif name == 'tether':
                    planned_path[name].append(to_numpy_flat(subspace_state, self.n_tether_state))

        # now convert lists to arrays
        planned_path_np = {}
        for k, v in planned_path.items():
            planned_path_np[k] = np.array(v)

        np_controls = np.ndarray((ompl_path.getControlCount(), self.n_control))
        for time_idx, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
            # duration is always be 1 for control::RRT, not so for control::SST
            np_controls[time_idx] = ompl_control_to_model_action(control, self.n_control).squeeze()

        return np_controls, planned_path_np, full_env_data, solved


def interpret_planner_status(planner_status: ob.PlannerStatus, verbose: int = 0):
    if verbose >= 1:
        # If the planner failed, print the error
        if not planner_status:
            print(Fore.RED + planner_status.asString() + Fore.RESET)
