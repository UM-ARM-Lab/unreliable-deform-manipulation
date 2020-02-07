from typing import Dict

import ompl.control as oc

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.viz_object import VizObject
from state_space_dynamics.base_forward_model import BaseForwardModel


class NearestRRT(MyPlanner):

    def __init__(self,
                 fwd_model: BaseForwardModel,
                 classifier_model: BaseClassifier,
                 planner_params: Dict,
                 services: GazeboServices,
                 viz_object: VizObject,
                 seed: int):
        super().__init__(fwd_model,
                         classifier_model,
                         planner_params,
                         services,
                         viz_object,
                         seed)

        self.planner = oc.RRT(self.si)
        self.planner.setIntermediateStates(True)  # this is necessary, because we use this to generate datasets
        self.ss.setPlanner(self.planner)
        self.si.setPropagationStepSize(self.fwd_model.dt)
        self.si.setMinMaxControlDuration(1, 50)
