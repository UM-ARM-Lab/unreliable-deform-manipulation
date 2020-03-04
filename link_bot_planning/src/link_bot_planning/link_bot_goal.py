from typing import Dict

import ompl.base as ob
import ompl.control as oc

from link_bot_planning.planning_scenario import PlanningScenario
from link_bot_planning.state_spaces import from_numpy, compound_to_numpy
from link_bot_planning.viz_object import VizObject


class MyGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 threshold: float,
                 goal,
                 viz: VizObject,
                 planning_scenario: PlanningScenario,
                 state_description: Dict):
        super(MyGoalRegion, self).__init__(si)
        self.goal = goal
        self.setThreshold(threshold)
        self.viz = viz
        self.planning_scenario = planning_scenario
        self.state_description = state_description

    def distanceGoal(self, state: ob.CompoundStateInternal):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        named_states = compound_to_numpy(self.state_description, state)
        distance = self.planning_scenario.distance_to_goal(state=named_states,
                                                           goal=self.goal)
        return distance

    def sampleGoal(self, state_out: ob.CompoundStateInternal):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random valid rope configuration via the state space sampler
        sampler.sampleUniform(state_out)

        named_states = compound_to_numpy(self.state_description, state_out)
        goal_state = self.planning_scenario.sample_goal(state=named_states,
                                                        goal=self.goal)

        # Only sets the link_bot subspace, because that's all we care about in this planning scenario
        for subspace_name, state in goal_state.items():
            subspace_description = self.state_description[subspace_name]
            subspace_idx = subspace_description['idx']
            n_state = subspace_description['n_state']
            from_numpy(state, state_out[subspace_idx], n_state)

        self.viz.states_sampled_at.append(goal_state)

    def maxSampleCount(self):
        return 100
