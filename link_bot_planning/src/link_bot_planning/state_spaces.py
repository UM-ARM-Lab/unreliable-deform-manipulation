from typing import List

import numpy as np
from ompl import base as ob
from ompl import control as oc

from link_bot_pycommon import link_bot_pycommon


class ValidRopeConfigurationSampler(ob.RealVectorStateSampler):

    def __init__(self, state_space, extent: List[float], link_length: float):
        super(ValidRopeConfigurationSampler, self).__init__(state_space)
        self.extent = extent
        self.link_length = link_length

    def sampleUniform(self, state_out: ob.State):
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

    def sampleUniform(self, state_out: ob.State):
        # passing 0 length will make it possible the sample things out of the bounds of the arena
        random_rope_configuration = link_bot_pycommon.make_random_rope_configuration(self.extent, length=self.link_length)
        state_out[0][0] = random_rope_configuration[0]
        state_out[0][1] = random_rope_configuration[1]
        state_out[0][2] = random_rope_configuration[2]
        state_out[0][3] = random_rope_configuration[3]
        state_out[0][4] = random_rope_configuration[4]
        state_out[0][5] = random_rope_configuration[5]
        # FIXME:
        state_out[1] = np.random.uniform(0, 1, 100 * 100)


def to_numpy(state_or_control, dim):
    np_state_or_control = np.ndarray((1, dim))
    for i in range(dim):
        np_state_or_control[0, i] = state_or_control[i]
    return np_state_or_control


def from_numpy(np_state_or_control, out, dim):
    for i in range(dim):
        out[i] = np_state_or_control[0, i]


def from_numpy_compound(np_state, local_sdf, compound_start, state_dim):
    for i in range(state_dim):
        compound_start[0][i] = np_state[0, i]
    for i, sdf_value in enumerate(local_sdf.sdf.flatten().astype(np.float64)):
        compound_start[1][i] = sdf_value
    print(compound_start[1])
