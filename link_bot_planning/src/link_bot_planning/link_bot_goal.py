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

    def __init__(self, si, threshold, numpy_goal, viz: VizObject, subspace_idx: int, point_idx: int, n_state: int):
        super(LinkBotCompoundGoal, self).__init__(si)
        self.goal_x = numpy_goal[0]
        self.goal_y = numpy_goal[1]
        self.setThreshold(threshold)
        self.viz = viz
        self.n_state = n_state
        self.point_idx = point_idx
        self.subspace_idx = subspace_idx
        self.n_links = link_bot_pycommon.n_state_to_n_links(self.n_state)
        self.x_idx = 2 * self.point_idx
        self.y_idx = 2 * self.point_idx + 1

    def distanceGoal(self, state: ob.CompoundStateInternal):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        point_x = state[self.subspace_idx][self.x_idx]
        point_y = state[self.subspace_idx][self.y_idx]
        dtg = np.linalg.norm([point_x - self.goal_x, point_y - self.goal_y])
        return dtg

    def sampleGoal(self, state_out: ob.CompoundStateInternal):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random valid rope configuration
        sampler.sampleUniform(state_out)

        # translate it so that the tail is at the goal
        np_s = to_numpy(state_out[self.subspace_idx], self.n_state)
        np_points = np.reshape(np_s, [-1, 2])
        np_points -= np_points[self.point_idx]
        np_points += np.array([self.goal_x, self.goal_y])

        state_out[self.subspace_idx][self.x_idx] = self.goal_x
        state_out[self.subspace_idx][self.y_idx] = self.goal_y

        self.viz.states_sampled_at.append(to_numpy(state_out[self.subspace_idx], self.n_state))
        # TODO: test this
        import ipdb; ipdb.set_trace()

    def maxSampleCount(self):
        return 100
