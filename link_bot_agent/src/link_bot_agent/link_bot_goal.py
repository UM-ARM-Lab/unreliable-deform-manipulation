import ompl.base as ob
import numpy as np


class LinkBotGoal(ob.GoalRegion):

    def __init__(self, si, threshold, numpy_goal):
        super(LinkBotGoal, self).__init__(si)
        self.tail_x = numpy_goal[0, 0]
        self.tail_y = numpy_goal[1, 0]
        self.setThreshold(threshold)

    def distanceGoal(self, state):
        return np.linalg.norm([state[0] - self.tail_x, state[1] - self.tail_y])
