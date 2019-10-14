from typing import List

import numpy as np
from ompl import base as ob

from link_bot_pycommon import link_bot_pycommon


class ValidRopeConfigurationSampler(ob.RealVectorStateSampler):

    def __init__(self, state_space, extent: List[float], link_length: float):
        super(ValidRopeConfigurationSampler, self).__init__(state_space)
        self.extent = extent
        self.link_length = link_length

    def sampleUniform(self, state_out: ob.AbstractState):
        # passing 0 length will make it possible the sample things out of the bounds of the arena
        random_rope_configuration = link_bot_pycommon.make_random_rope_configuration(self.extent, length=self.link_length)
        state_out[0] = random_rope_configuration[0]
        state_out[1] = random_rope_configuration[1]
        state_out[2] = random_rope_configuration[2]
        state_out[3] = random_rope_configuration[3]
        state_out[4] = random_rope_configuration[4]
        state_out[5] = random_rope_configuration[5]


class ValidRopeConfigurationCompoundSampler(ob.RealVectorStateSampler):

    def __init__(self, state_space, extent: List[float], link_length: float):
        super(ValidRopeConfigurationCompoundSampler, self).__init__(state_space)
        self.extent = extent
        self.link_length = link_length

    def sampleUniform(self, state_out: ob.CompoundStateInternal):
        # passing 0 length will make it possible the sample things out of the bounds of the arena
        random_rope_configuration = link_bot_pycommon.make_random_rope_configuration(self.extent, length=self.link_length)
        state_out[0][0] = random_rope_configuration[0]
        state_out[0][1] = random_rope_configuration[1]
        state_out[0][2] = random_rope_configuration[2]
        state_out[0][3] = random_rope_configuration[3]
        state_out[0][4] = random_rope_configuration[4]
        state_out[0][5] = random_rope_configuration[5]


def to_numpy(state_or_control, dim):
    np_state_or_control = np.ndarray((1, dim))
    for i in range(dim):
        np_state_or_control[0, i] = state_or_control[i]
    return np_state_or_control


def to_numpy_local_env(sdf_state, h_rows, w_cols):
    np_sdf = np.ndarray((h_rows, w_cols))
    for r, c in np.ndindex(h_rows, w_cols):
        i = (h_rows * r) + c
        np_sdf[r, c] = sdf_state[i]
    return np_sdf


def from_numpy(np_state_or_control, out, dim):
    if np_state_or_control.ndim == 2:
        for i in range(dim):
            out[i] = np_state_or_control[0, i]
    else:
        for i in range(dim):
            out[i] = np_state_or_control[i]
