import pathlib
from typing import Tuple, List, Optional, Dict

import numpy as np
import ompl.base as ob
import ompl.control as oc
from colorama import Fore

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_gazebo.gazebo_utils import GazeboServices, get_local_occupancy_data
from link_bot_planning.link_bot_goal import LinkBotCompoundGoal
from link_bot_planning.random_directed_control_sampler import RandomDirectedControlSampler
from link_bot_planning.state_spaces import to_numpy, from_numpy, to_numpy_local_env, ValidRopeConfigurationCompoundSampler, \
    TrainingSetCompoundSampler
from link_bot_planning.viz_object import VizObject
from link_bot_pycommon import link_bot_sdf_utils, link_bot_pycommon
from state_space_dynamics.base_forward_model import BaseForwardModel


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
        self.full_env_params = self.fwd_model.full_env_params
        self.planner_params = planner_params
        self.services = services
        self.viz_object = viz_object
        self.si = ob.SpaceInformation(ob.StateSpace())
        self.planner = ob.Planner(self.si, 'PlaceholderPlanner')
        self.rope_length = fwd_model.hparams['dynamics_dataset_hparams']['rope_length']
        self.seed = seed

        self.state_space = ob.CompoundStateSpace()
        self.n_local_env = self.fwd_model.local_env_params.w_cols * self.fwd_model.local_env_params.h_rows
        self.local_env_space = ob.RealVectorStateSpace(self.n_local_env)
        epsilon = 1e-3
        self.local_env_space.setBounds(-epsilon, 1 + epsilon)

        self.local_env_origin_space = ob.RealVectorStateSpace(2)
        self.local_env_origin_space.setBounds(-10000, 10000)

        self.config_space = ob.RealVectorStateSpace(self.n_state)
        self.config_space_bounds = ob.RealVectorBounds(self.n_state)
        for i in range(self.n_state):
            if i % 2 == 0:
                self.config_space_bounds.setLow(i, -self.planner_params['w'] / 2)
                self.config_space_bounds.setHigh(i, self.planner_params['w'] / 2)
            else:
                self.config_space_bounds.setLow(i, -self.planner_params['h'] / 2)
                self.config_space_bounds.setHigh(i, self.planner_params['h'] / 2)
        self.config_space.setBounds(self.config_space_bounds)

        # the rope is just 6 real numbers with no bounds
        # by setting the weight to 1, it means that distances are based only on the rope config not the local environment
        # so when we sample a state, we get a random local environment, but the nearest neighbor is based only on the rope config
        # this is sort of a specialization, but I think it's justified. Otherwise nothing would work I suspect (but I didn't test)
        self.state_space.addSubspace(self.config_space, weight=self.planner_params['subspace_weights'][0])
        # the local environment is a rows*cols flat vector of numbers from 0 to 1
        self.state_space.addSubspace(self.local_env_space, weight=self.planner_params['subspace_weights'][1])
        # origin
        self.state_space.addSubspace(self.local_env_origin_space, weight=self.planner_params['subspace_weights'][2])

        self.state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(self.state_sampler_allocator))

        control_bounds = ob.RealVectorBounds(2)
        control_bounds.setLow(-self.planner_params['max_v'])
        control_bounds.setHigh(self.planner_params['max_v'])
        self.control_space = oc.RealVectorControlSpace(self.state_space, self.n_control)
        self.control_space.setBounds(control_bounds)

        self.ss = oc.SimpleSetup(self.control_space)

        self.si = self.ss.getSpaceInformation()

        self.ss.setStatePropagator(oc.StatePropagatorFn(self.propagate))
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_valid))

        if planner_params['directed_control_sampler'] == 'simple':
            pass  # the default
        elif planner_params['directed_control_sampler'] == 'random':
            self.si.setDirectedControlSamplerAllocator(RandomDirectedControlSampler.allocator())

        self.full_envs = None
        self.full_env_orgins = None

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

    def propagate(self, start, control, duration, state_out):
        del duration  # unused, multi-step propagation is handled inside propagateWhileValid
        np_s = to_numpy(start[0], self.n_state)
        np_u = np.expand_dims(to_numpy(control, self.n_control), axis=0)
        local_env_data = self.get_local_env_at(np_s[0, -2], np_s[0, -1])

        # if self.viz_object.new_sample:
        #     print(np_s[0, 0], np_s[0, 1])  # this should be the nearest neighbor
        #     self.viz_object.new_sample = False
        #
        #     plt.figure()
        #     plt.title("sampled local env")
        #     plt.imshow(np.flipud(self.viz_object.debugging1))
        #     plt.figure()
        #     plt.title("nearest neighbor in search tree")
        #     plt.imshow(local_env_data.image)
        #     plt.show()
        #     input()
        #     # ipdb.set_trace()

        # use the forward model to predict the next configuration
        points_next = self.fwd_model.predict(full_envs=self.full_envs,
                                             full_env_origins=self.full_env_origins,
                                             resolution_s=np.array([[self.fwd_model.full_env_params.res]]),
                                             state=np_s,
                                             actions=np_u)
        np_s_next = points_next[:, 1].reshape([1, self.n_state])

        # validate the edge
        accept_probabilities = self.classifier_model.predict(local_env_data=[local_env_data], s1=np_s, s2=np_s_next, action=np_u)
        accept_probability = accept_probabilities[0]
        random_accept = np.random.uniform(0, 1) <= self.planner_params['random_epsilon']
        classifier_accept = np.random.uniform(0, 1) <= accept_probability
        edge_is_valid = classifier_accept or random_accept

        # copy the result into the ompl state data structure
        if not edge_is_valid:
            # This will ensure this edge is not added to the tree
            self.viz_object.rejected_samples.append(np_s_next[0])
            state_out[0][0] = 1000
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
    def plan(self, np_start: np.ndarray, tail_goal_point: np.ndarray, full_env_data: link_bot_sdf_utils.OccupancyData) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[List[link_bot_sdf_utils.OccupancyData]],
        Optional[link_bot_sdf_utils.OccupancyData],
        ob.PlannerStatus]:
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

            np_states = np.ndarray((ompl_path.getStateCount(), self.n_state))
            np_controls = np.ndarray((ompl_path.getControlCount(), self.n_control))
            planner_local_envs = []
            for i, state in enumerate(ompl_path.getStates()):
                np_s = to_numpy(state[0], self.n_state)
                np_states[i] = np_s
                grid = to_numpy_local_env(state[1], self.fwd_model.local_env_params.h_rows,
                                          self.fwd_model.local_env_params.w_cols)
                res_2d = np.array([self.fwd_model.local_env_params.res, self.fwd_model.local_env_params.res])
                origin = to_numpy(state[2], 2)[0]
                planner_local_env = link_bot_sdf_utils.OccupancyData(grid, res_2d, origin)
                planner_local_envs.append(planner_local_env)
            for i, (control, duration) in enumerate(zip(ompl_path.getControls(), ompl_path.getControlDurations())):
                # duration is always be 1 for control::RRT, not so for control::SST
                np_controls[i] = to_numpy(control, self.n_control)

            return np_controls, np_states, planner_local_envs, full_env_data, solved
        return None, None, [], full_env_data, solved

    def state_sampler_allocator(self, state_space):
        if self.planner_params['sampler_type'] == 'random':
            extent = [-self.planner_params['w'] / 2,
                      self.planner_params['w'] / 2,
                      -self.planner_params['h'] / 2,
                      self.planner_params['h'] / 2]
            sampler = ValidRopeConfigurationCompoundSampler(state_space,
                                                            self.viz_object,
                                                            extent=extent,
                                                            n_state=self.n_state,
                                                            rope_length=self.rope_length,
                                                            max_angle_rad=self.planner_params['max_angle_rad'])
        elif self.planner_params['sampler_type'] == 'sample_train':
            sampler = TrainingSetCompoundSampler(state_space,
                                                 self.viz_object,
                                                 train_dataset=self.training_dataset,
                                                 sequence_length=self.train_dataset_max_sequence_length)
        else:
            raise ValueError("Invalid sampler type {}".format(self.planner_params['sampler_type']))

        return sampler


def interpret_planner_status(planner_status: ob.PlannerStatus, verbose: int = 0):
    if verbose >= 1:
        # If the planner failed, print the error
        if not planner_status:
            print(Fore.RED + planner_status.asString() + Fore.RESET)
