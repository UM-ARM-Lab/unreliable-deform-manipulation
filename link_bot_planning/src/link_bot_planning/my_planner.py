from typing import Tuple, List

import numpy as np
import ompl.base as ob

from link_bot_classifiers.base_classifier import BaseClassifier
from state_space_dynamics.base_forward_model import BaseForwardModel
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning.viz_object import VizObject
from link_bot_planning.params import PlannerParams, EnvParams
from link_bot_pycommon import link_bot_sdf_utils


class MyPlanner:

    def __init__(self,
                 fwd_model: BaseForwardModel,
                 classifier_model: BaseClassifier,
                 planner_params: PlannerParams,
                 env_params: EnvParams,
                 services: GazeboServices,
                 viz_object: VizObject):
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.n_state = self.fwd_model.n_state
        self.n_control = self.fwd_model.n_control
        self.env_params = env_params
        self.planner_params = planner_params
        self.services = services
        self.viz_object = viz_object
        self.si = ob.SpaceInformation(ob.StateSpace())
        self.planner = ob.Planner(self.si, 'PlaceholderPlanner')

    def plan(self, np_start: np.ndarray,
             tail_goal_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[link_bot_sdf_utils.OccupancyData]]:
        pass
