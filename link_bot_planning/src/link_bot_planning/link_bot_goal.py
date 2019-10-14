import numpy as np
import ompl.base as ob


class LinkBotGoal(ob.GoalSampleableRegion):

    def __init__(self, si, threshold, numpy_goal):
        super(LinkBotGoal, self).__init__(si)
        self.tail_x = numpy_goal[0]
        self.tail_y = numpy_goal[1]
        self.setThreshold(threshold)

    def distanceGoal(self, state):
        return np.linalg.norm([state[0] - self.tail_x, state[1] - self.tail_y])

    def sampleGoal(self, state_out):
        sampler = self.getSpaceInformation().allocStateSampler()
        sampler.sampleUniform(state_out)
        state_out[2] = (state_out[2] - state_out[0]) + self.tail_x
        state_out[3] = (state_out[3] - state_out[1]) + self.tail_y
        state_out[4] = (state_out[4] - state_out[0]) + self.tail_x
        state_out[5] = (state_out[5] - state_out[1]) + self.tail_y
        state_out[0] = self.tail_x
        state_out[1] = self.tail_y

    def maxSampleCount(self):
        return 100


class LinkBotCompoundGoal(ob.GoalSampleableRegion):

    def __init__(self, si, threshold, numpy_goal):
        super(LinkBotCompoundGoal, self).__init__(si)
        self.tail_x = numpy_goal[0]
        self.tail_y = numpy_goal[1]
        self.setThreshold(threshold)

    def distanceGoal(self, state: ob.CompoundStateInternal):
        """
        Uses the distance between the tail point and the goal point
        """
        return np.linalg.norm([state[0][0] - self.tail_x, state[0][1] - self.tail_y])

    def sampleGoal(self, state_out: ob.CompoundStateInternal):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sampe a random valid rope configuration
        sampler.sampleUniform(state_out)
        # translate it so that the tail is at the goal
        state_out[0][2] = (state_out[0][2] - state_out[0][0]) + self.tail_x
        state_out[0][3] = (state_out[0][3] - state_out[0][1]) + self.tail_y
        state_out[0][4] = (state_out[0][4] - state_out[0][0]) + self.tail_x
        state_out[0][5] = (state_out[0][5] - state_out[0][1]) + self.tail_y
        state_out[0][0] = self.tail_x
        state_out[0][1] = self.tail_y

    def maxSampleCount(self):
        return 100
