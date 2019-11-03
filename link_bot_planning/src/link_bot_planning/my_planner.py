from typing import Tuple, List

import numpy as np
import ompl.base as ob

from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning.ompl_viz import VizObject
from link_bot_planning.params import LocalEnvParams, PlannerParams, EnvParams
from link_bot_pycommon import link_bot_sdf_utils


class MyPlanner:

    def __init__(self,
                 fwd_model,
                 classifier_model,
                 dt: float,
                 planner_params: PlannerParams,
                 local_env_params: LocalEnvParams,
                 env_params: EnvParams,
                 services: GazeboServices,
                 viz_object: VizObject):
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.dt = dt
        self.n_state = self.fwd_model.n_state
        self.n_control = self.fwd_model.n_control
        self.local_env_params = local_env_params
        self.env_params = env_params
        self.planner_params = planner_params
        self.services = services
        self.viz_object = viz_object
        self.si = ob.SpaceInformation(ob.StateSpace())
        self.planner = ob.Planner(self.si, 'PlaceholderPlanner')

    def plan(self, np_start: np.ndarray,
             tail_goal_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[link_bot_sdf_utils.OccupancyData]]:
        pass
