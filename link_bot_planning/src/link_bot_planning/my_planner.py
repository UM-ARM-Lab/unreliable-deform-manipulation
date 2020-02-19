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
from link_bot_pycommon.link_bot_pycommon import print_dict
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
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
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
                for i in range(self.n_tether_state):
                    if i % 2 == 0:
                        self.tether_space_bounds.setLow(i, -self.planner_params['w'] / 2)
                        self.tether_space_bounds.setHigh(i, self.planner_params['w'] / 2)
                    else:
                        self.tether_space_bounds.setLow(i, -self.planner_params['h'] / 2)
                        self.tether_space_bounds.setHigh(i, self.planner_params['h'] / 2)
                self.tether_space.setBounds(self.tether_space_bounds)
                self.state_space.addSubspace(self.tether_space, weight=component_description['weight'])

        self.state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(self.state_sampler_allocator))

        control_bounds = ob.RealVectorBounds(2)
        control_bounds.setLow(0, -np.pi)
        control_bounds.setHigh(0, np.pi)
        control_bounds.setLow(1, 0)
        max_delta_pos = ros_pycommon.get_max_speed() * self.fwd_model.dt * 0.9  # safety factor to make planning more accurate
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

        self.full_envs = None
        self.full_env_origins = None

        if planner_params['sampler_type'] == 'sample_train':
            self.dataset_dirs = [pathlib.Path(p) for p in self.fwd_model.hparams['datasets']]
            dataset = LinkBotStateSpaceDataset(self.dataset_dirs)
            self.training_dataset = dataset.get_datasets(mode='train',
                                                         shuffle=True,
                                                         seed=self.seed,
                                                         sequence_length=None,
                                                         batch_size=None)  # no batching
            self.train_dataset_max_sequence_length = dataset.max_sequence_length

    def get_local_env_at(self, x: float, y: float):
        center_point = np.array([x, y])
        return get_local_occupancy_data(rows=self.fwd_model.local_env_params.h_rows,
                                        cols=self.fwd_model.local_env_params.w_cols,
                                        res=self.fwd_model.local_env_params.res,
                                        center_point=center_point,
                                        services=self.services)

    def is_valid(self, state):
        return self.state_space.getSubspace(0).satisfiesBounds(state[0])

    def edge_is_valid(self, np_s: np.ndarray, np_u: np.ndarray, np_s_next: np.ndarray):
        local_env_data = self.get_local_env_at(np_s[0, -2], np_s[0, -1])

        if self.classifier_model.model_hparams['image_type'] == 'transition_image':
        elif self.classifier_model.model_hparams['image_type'] == 'trajectory_image':
            # Get the whole path?
        else:
            raise ValueError()

        accept_probabilities = self.classifier_model.predict(local_env_data=[local_env_data], s1=np_s, s2=np_s_next, action=np_u)

        accept_probability = accept_probabilities[0]
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
        np_s = to_numpy(start[0], self.n_state)

        np_u = ompl_control_to_model_action(control, self.n_control)

        # use the forward model to predict the next configuration
        points_next = self.fwd_model.predict(full_envs=self.full_envs,
                                             full_env_origins=self.full_env_origins,
                                             resolution_s=np.array([[self.fwd_model.full_env_params.res]]),
                                             state=np_s,
                                             actions=np_u)
        np_s_next = points_next[:, 1].reshape([1, self.n_state])

        # validate the edge
        edge_is_valid = self.edge_is_valid(np_s, np_u, np_s_next)

        # copy the result into the ompl state data structure
        if not edge_is_valid:
            # This will ensure this edge is not added to the tree
            state_out[0][0] = 1000
            self.viz_object.rejected_samples.append(np_s_next[0])
        else:
            from_numpy(np_s_next, state_out[0], self.n_state)
            local_env_data = self.get_local_env_at(np_s_next[0, -2], np_s_next[0, -1])
            local_env = local_env_data.data.flatten().astype(np.float64)
            for idx in range(self.n_local_env):
                occupancy_value = local_env[idx]
                state_out[1][idx] = occupancy_value
            origin = local_env_data.origin.astype(np.float64)
            from_numpy(origin, state_out[2], 2)

    # TODO: make this return a data structure. something with a name
    def plan(self, np_start: np.ndarray, tail_goal_point: np.ndarray, full_env_data: link_bot_sdf_utils.OccupancyData):
        """
        :param full_env_data:
        :param np_start: 1 by n matrix
        :param tail_goal_point:  1 by n matrix
        :return: controls, states
        """
        self.full_envs = np.array([full_env_data.data])
        self.full_env_origins = np.array([full_env_data.origin])

        # create start and goal states
        start_local_occupancy = self.get_local_env_at(np_start[0, -2], np_start[0, -1])
        compound_start = ob.CompoundState(self.state_space)
        for i in range(self.n_state):
            compound_start()[0][i] = np_start[0, i]
        start_local_occupancy_flat_double = start_local_occupancy.data.flatten().astype(np.float64)
        for idx in range(self.n_local_env):
            occupancy_value = start_local_occupancy_flat_double[idx]
            compound_start()[1][idx] = occupancy_value
        start_local_occupancy_origin_double = start_local_occupancy.origin.astype(np.float64)
        compound_start()[2][0] = start_local_occupancy_origin_double[0]
        compound_start()[2][1] = start_local_occupancy_origin_double[1]

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
        return None, None, [], full_env_data, solved

    def state_sampler_allocator(self, state_space):
        if self.planner_params['sampler_type'] == 'random':
            extent = [-self.planner_params['w'] / 2,
                      self.planner_params['w'] / 2,
                      -self.planner_params['h'] / 2,
                      self.planner_params['h'] / 2]
            sampler = ValidRopeConfigurationCompoundSampler(state_space,
                                                            my_planner=self,
                                                            viz_object=self.viz_object,
                                                            extent=extent,
                                                            n_state=self.n_state,
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
