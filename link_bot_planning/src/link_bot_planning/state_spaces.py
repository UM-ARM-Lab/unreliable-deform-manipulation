from typing import List, Dict

import numpy as np
import tensorflow as tf
from ompl import base as ob

from link_bot_planning.viz_object import VizObject
from link_bot_pycommon import link_bot_pycommon


class ValidRopeConfigurationSampler(ob.RealVectorStateSampler):

    def __init__(self,
                 state_space,
                 viz_object: VizObject,
                 extent: List[float],
                 n_state: int,
                 rope_length: float,
                 max_angle_rad: float,
                 rng: np.random.RandomState):
        super(ValidRopeConfigurationSampler, self).__init__(state_space)
        self.extent = extent
        self.rope_length = rope_length
        self.n_links = link_bot_pycommon.n_state_to_n_links(n_state)
        self.n_state = n_state
        if self.n_links == 0:
            self.link_length = 0
        else:
            self.link_length = rope_length / self.n_links
        self.viz_object = viz_object
        self.max_angle_rad = max_angle_rad
        self.rng = rng

    def sampleUniform(self, state_out: ob.AbstractState):
        random_rope_configuration = link_bot_pycommon.make_random_rope_configuration(self.extent,
                                                                                     n_state=self.n_state,
                                                                                     link_length=self.link_length,
                                                                                     max_angle_rad=self.max_angle_rad,
                                                                                     rng=self.rng)
        for i in range(random_rope_configuration.shape[0]):
            state_out[i] = random_rope_configuration[i]
        self.viz_object.states_sampled_at.append(random_rope_configuration)


class ValidRopeConfigurationCompoundSampler(ob.RealVectorStateSampler):

    def __init__(self,
                 state_space,
                 my_planner,
                 viz_object: VizObject,
                 extent: List[float],
                 n_rope_state: int,
                 subspace_name_to_index: Dict[str, int],
                 rope_length: float,
                 max_angle_rad: float,
                 rng: np.random.RandomState, ):
        super(ValidRopeConfigurationCompoundSampler, self).__init__(state_space)
        self.my_planner = my_planner
        self.extent = extent
        self.rope_length = rope_length
        self.n_rope_state = n_rope_state
        self.subspace_name_to_index = subspace_name_to_index
        self.link_bot_subspace_idx = self.subspace_name_to_index['link_bot']
        self.n_links = link_bot_pycommon.n_state_to_n_links(self.n_rope_state)
        if self.n_links == 0:
            self.link_length = 0
        else:
            self.link_length = rope_length / self.n_links
        self.viz_object = viz_object
        self.max_angle_rad = max_angle_rad
        self.rng = rng

    def sampleUniform(self, state_out: ob.CompoundStateInternal):
        # We don't bother filling out the other components of state here because we assume they have zero weight
        # so distance calculations won't consider them
        random_rope_configuration = link_bot_pycommon.make_random_rope_configuration(self.extent,
                                                                                     n_state=self.n_rope_state,
                                                                                     link_length=self.link_length,
                                                                                     max_angle_rad=self.max_angle_rad,
                                                                                     rng=self.rng)

        from_numpy(random_rope_configuration, state_out[self.link_bot_subspace_idx], random_rope_configuration.shape[0])

        self.viz_object.states_sampled_at.append(random_rope_configuration)


class TrainingSetCompoundSampler(ob.RealVectorStateSampler):

    def __init__(self,
                 state_space,
                 viz_object: VizObject,
                 train_dataset: tf.data.Dataset,
                 sequence_length: int,
                 rng: np.random.RandomState):
        super(TrainingSetCompoundSampler, self).__init__(state_space)
        self.viz_object = viz_object
        self.infinite_dataset = train_dataset.repeat()  # infinite!
        self.iter = iter(self.infinite_dataset)
        self.sequence_length = sequence_length
        self.rng = rng

    def sampleUniform(self, state_out: ob.CompoundStateInternal):
        """
        :param state_out: only the rope config subspace [0] matters, the others are overwritten in the propagate function
         and in the case of where we are using the free-space model, the local env gets ignored anyways
        :return:
        """
        example, _ = next(self.iter)

        # since the training data stores sequences, we have to randomly sample a specific time step
        t = self.rng.randint(0, self.sequence_length - 1)  # -1 because the last time step is in the output, not the input
        state = example['state_s'][t].numpy()
        local_env = example['actual_local_env_s/env'][t].numpy()
        local_env_origin = example['actual_local_env_s/origin'][t].numpy()

        for i, s_i in enumerate(state):
            state_out[0][i] = np.float64(s_i)

        for i, env_i in enumerate(local_env.flatten()):
            state_out[1][i] = np.float64(env_i)

        state_out[2][0] = np.float64(local_env_origin[0])
        state_out[2][1] = np.float64(local_env_origin[1])
        # NOTE: it doesn't really make sense to visualize the rope configurations when sampling in this way,
        #  because the rope configurations cannot be viewed in isolation, you'd need to also see the local env
        # self.viz_object.states_sampled_at.append(state)
        # self.viz_object.debugging1 = local_env
        # self.viz_object.new_sample = True


def to_numpy_flat(state_or_control, dim: int):
    np_state_or_control = np.ndarray(dim)
    for i in range(dim):
        np_state_or_control[i] = state_or_control[i]
    return np_state_or_control


def to_numpy(state_or_control, dim: int):
    np_state_or_control = np.ndarray((1, dim))
    for i in range(dim):
        np_state_or_control[0, i] = state_or_control[i]
    return np_state_or_control


def to_numpy_local_env(local_env_state: ob.AbstractState, h_rows: int, w_cols: int):
    np_local_env = np.ndarray((h_rows, w_cols))
    for r, c in np.ndindex(h_rows, w_cols):
        i = (h_rows * r) + c
        np_local_env[r, c] = local_env_state[i]
    return np_local_env


def from_numpy(np_state_or_control: np.ndarray,
               out,
               dim: int):
    if np_state_or_control.ndim == 2:
        for i in range(dim):
            out[i] = np_state_or_control[0, i]
    else:
        for i in range(dim):
            out[i] = np_state_or_control[i]
