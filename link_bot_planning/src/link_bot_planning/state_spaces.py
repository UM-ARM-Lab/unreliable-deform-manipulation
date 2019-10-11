import numpy as np
from ompl import base as ob
from ompl import control as oc

from link_bot_pycommon import link_bot_pycommon


class LinkBotControlSpace(oc.RealVectorControlSpace):

    def __init__(self, state_space, n_control):
        super(LinkBotControlSpace, self).__init__(state_space, n_control)


class MinimalLinkBotStateSpace(ob.RealVectorStateSpace):

    def __init__(self, n_state):
        super(MinimalLinkBotStateSpace, self).__init__(n_state)
        self.setDimensionName(0, 'tail_x')
        self.setDimensionName(1, 'tail_y')
        self.setDimensionName(2, 'theta_0')
        self.setDimensionName(3, 'theta_1')


class LinkBotStateSpaceSampler(ob.RealVectorStateSampler):

    def __init__(self, state_space, extent):
        super(LinkBotStateSpaceSampler, self).__init__(state_space)
        self.extent = extent

    def sampleUniform(self, state_out):
        random_rope_configuration = link_bot_pycommon.make_random_rope_configuration(self.extent)
        state_out[0] = random_rope_configuration[0]
        state_out[1] = random_rope_configuration[1]
        state_out[2] = random_rope_configuration[2]
        state_out[3] = random_rope_configuration[3]
        state_out[4] = random_rope_configuration[4]
        state_out[5] = random_rope_configuration[5]


class LinkBotStateSpace(ob.RealVectorStateSpace):

    def __init__(self, n_state, extent):
        super(LinkBotStateSpace, self).__init__(n_state)
        self.extent = extent
        self.setDimensionName(0, 'tail_x')
        self.setDimensionName(1, 'tail_y')
        self.setDimensionName(2, 'mid_x')
        self.setDimensionName(3, 'mid_y')

    def allocator(self, state_space):
        sampler = LinkBotStateSpaceSampler(state_space, self.extent)
        return sampler


class TailStateSpace(ob.RealVectorStateSpace):

    def __init__(self, n_state, extent):
        super(TailStateSpace, self).__init__(n_state)
        self.extent = extent
        self.setDimensionName(0, 'tail_x')
        self.setDimensionName(1, 'tail_y')

    def allocator(self, state_space):
        sampler = ValidRopeConfigurationSampler(state_space, self.extent)
        return sampler

    # def allocState(self):
    #     print("ALLOC TAIL STATE SPACE")
    #     return super(TailStateSpace, self).allocState()
    #
    # def freeState(self, state):
    #     print("FREE TAIL STATE SPACE")
    #     super(TailStateSpace, self).freeState(state)


class ValidRopeConfigurationSampler(ob.RealVectorStateSampler):

    def __init__(self, state_space, extent, link_length):
        super(ValidRopeConfigurationSampler, self).__init__(state_space)
        self.extent = extent
        self.link_length

    def sampleUniform(self, state_out):
        # passing 0 length will make it possible the sample things out of the bounds of the arena
        random_rope_configuration = link_bot_pycommon.make_random_rope_configuration(self.extent, length=self.link_length)
        state_out[0] = random_rope_configuration[0]
        state_out[1] = random_rope_configuration[1]
        state_out[2] = random_rope_configuration[2]
        state_out[3] = random_rope_configuration[3]
        state_out[4] = random_rope_configuration[4]
        state_out[5] = random_rope_configuration[5]


def to_numpy(state_or_control, dim):
    np_state_or_control = np.ndarray((1, dim))
    for i in range(dim):
        np_state_or_control[0, i] = state_or_control[i]
    return np_state_or_control


def from_numpy(np_state_or_control, out, dim):
    for i in range(dim):
        out[i] = np_state_or_control[0, i]
