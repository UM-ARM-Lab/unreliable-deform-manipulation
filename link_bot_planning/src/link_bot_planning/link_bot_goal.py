import numpy as np
import ompl.base as ob

from link_bot_planning.state_spaces import to_numpy
from link_bot_planning.viz_object import VizObject
from link_bot_pycommon import link_bot_pycommon


class LinkBotGoal(ob.GoalSampleableRegion):

    def __init__(self, si, threshold, numpy_goal, viz: VizObject, n_state: int):
        super(LinkBotGoal, self).__init__(si)
        self.tail_x = numpy_goal[0]
        self.tail_y = numpy_goal[1]
        self.setThreshold(threshold)
        self.viz = viz
        self.n_state = n_state
        self.n_links = link_bot_pycommon.n_state_to_n_links(self.n_state)

    def distanceGoal(self, state):
        return np.linalg.norm([state[0] - self.tail_x, state[1] - self.tail_y])

    def sampleGoal(self, state_out):
        sampler = self.getSpaceInformation().allocStateSampler()
        sampler.sampleUniform(state_out)
        for i in range(0, self.n_links):
            state_out[2 * i + 2] = (state_out[2 * i + 2] - state_out[0]) + self.tail_x
            state_out[2 * i + 3] = (state_out[2 * i + 3] - state_out[1]) + self.tail_y
        state_out[0] = self.tail_x
        state_out[1] = self.tail_y

        self.viz.states_sampled_at.append(to_numpy(state_out, self.n_state))

    def maxSampleCount(self):
        return 100


class LinkBotCompoundGoal(ob.GoalSampleableRegion):

    def __init__(self, si, threshold, numpy_goal, viz: VizObject, n_state: int):
        super(LinkBotCompoundGoal, self).__init__(si)
        self.tail_x = numpy_goal[0]
        self.tail_y = numpy_goal[1]
        self.setThreshold(threshold)
        self.viz = viz
        self.n_state = n_state
        self.n_links = link_bot_pycommon.n_state_to_n_links(self.n_state)
        # TODO: add these to the viz object

    def distanceGoal(self, state: ob.CompoundStateInternal):
        """
        Uses the distance between the tail point and the goal point
        """
        dtg = np.linalg.norm([state[0][0] - self.tail_x, state[0][1] - self.tail_y])
        return dtg

    def sampleGoal(self, state_out: ob.CompoundStateInternal):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random valid rope configuration
        sampler.sampleUniform(state_out)

        # translate it so that the tail is at the goal
        for i in range(0, self.n_links):
            state_out[0][2 * i + 2] = (state_out[0][2 * i + 2] - state_out[0][0]) + self.tail_x
            state_out[0][2 * i + 3] = (state_out[0][2 * i + 3] - state_out[0][1]) + self.tail_y

        state_out[0][0] = self.tail_x
        state_out[0][1] = self.tail_y

        self.viz.states_sampled_at.append(to_numpy(state_out[0], self.n_state))

    def maxSampleCount(self):
        return 100
