import numpy as np
from ompl import base as ob
from ompl import control as oc

from link_bot_pycommon import link_bot_pycommon


class LinkBotControlSpace(oc.RealVectorControlSpace):

    def __init__(self, state_space, n_control):
        super(LinkBotControlSpace, self).__init__(state_space, n_control)

    @staticmethod
    def to_numpy(control):
        assert isinstance(control, oc.RealVectorControlSpace.ControlType)
        np_u = np.ndarray((1, 2))
        np_u[0, 0] = control[0]
        np_u[0, 1] = control[1]
        return np_u

    @staticmethod
    def from_numpy(u, control_out):
        control_out[0] = u[0, 0]
        control_out[1] = u[0, 1]


class MinimalLinkBotStateSpace(ob.RealVectorStateSpace):

    def __init__(self, n_state):
        super(MinimalLinkBotStateSpace, self).__init__(n_state)
        self.setDimensionName(0, 'tail_x')
        self.setDimensionName(1, 'tail_y')
        self.setDimensionName(2, 'theta_0')
        self.setDimensionName(3, 'theta_1')

    @staticmethod
    def to_numpy(state, l=0.5):
        np_s = np.ndarray((1, 6))
        np_s[0, 0] = state[0]
        np_s[0, 1] = state[1]
        np_s[0, 2] = np_s[0, 0] + np.cos(state[2]) * l
        np_s[0, 3] = np_s[0, 1] + np.sin(state[2]) * l
        np_s[0, 4] = np_s[0, 2] + np.cos(state[3]) * l
        np_s[0, 5] = np_s[0, 3] + np.sin(state[3]) * l
        return np_s

    @staticmethod
    def from_numpy(np_s, state_out):
        state_out[0] = np_s[0, 0]
        state_out[1] = np_s[0, 1]
        state_out[2] = np.arctan2(np_s[0, 3] - np_s[0, 1], np_s[0, 2] - np_s[0, 0])
        state_out[3] = np.arctan2(np_s[0, 5] - np_s[0, 3], np_s[0, 4] - np_s[0, 2])


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
        self.setDimensionName(4, 'head_x')
        self.setDimensionName(5, 'head_y')

    def allocator(self, state_space):
        sampler = LinkBotStateSpaceSampler(state_space, self.extent)
        return sampler

    @staticmethod
    def to_numpy(state):
        np_s = np.ndarray((1, 6))
        for i in range(6):
            np_s[0, i] = state[i]
        return np_s

    @staticmethod
    def from_numpy(np_s, state_out):
        for i in range(6):
            state_out[i] = np_s[0, i]
