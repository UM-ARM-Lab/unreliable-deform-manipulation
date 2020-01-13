from typing import List

import numpy as np
from ompl import base as ob

from link_bot_planning.viz_object import VizObject
from link_bot_pycommon import link_bot_pycommon


class ValidRopeConfigurationSampler(ob.RealVectorStateSampler):

    def __init__(self, state_space, viz_object: VizObject, extent: List[float], n_state: int, rope_length: float):
        super(ValidRopeConfigurationSampler, self).__init__(state_space)
        self.extent = extent
        self.rope_length = rope_length
        self.n_links = int(n_state // 2 - 1)
        self.n_state = n_state
        self.link_length = rope_length / self.n_links
        self.viz_object = viz_object

    def sampleUniform(self, state_out: ob.AbstractState):
        random_rope_configuration = link_bot_pycommon.make_random_rope_configuration(self.extent,
                                                                                     n_state=self.n_state,
                                                                                     total_length=self.link_length)
        for i in range(random_rope_configuration.shape[0]):
            state_out[i] = random_rope_configuration[i]
        self.viz_object.states_sampled_at.append(random_rope_configuration)


class ValidRopeConfigurationCompoundSampler(ob.RealVectorStateSampler):

    def __init__(self, state_space, viz_object: VizObject, extent: List[float], n_state: int, rope_length: float):
        super(ValidRopeConfigurationCompoundSampler, self).__init__(state_space)
        self.extent = extent
        self.rope_length = rope_length
        self.n_links = int(n_state // 2 - 1)
        self.n_state = n_state
        self.link_length = rope_length / self.n_links
        self.viz_object = viz_object

    def sampleUniform(self, state_out: ob.CompoundStateInternal):
        random_rope_configuration = link_bot_pycommon.make_random_rope_configuration(self.extent,
                                                                                     n_state=self.n_state,
                                                                                     total_length=self.link_length)
        for i in range(random_rope_configuration.shape[0]):
            state_out[0][i] = random_rope_configuration[i]
        self.viz_object.states_sampled_at.append(random_rope_configuration)


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


def from_numpy(np_state_or_control: np.ndarray, out, dim: int):
    if np_state_or_control.ndim == 2:
        for i in range(dim):
            out[i] = np_state_or_control[0, i]
    else:
        for i in range(dim):
            out[i] = np_state_or_control[i]
