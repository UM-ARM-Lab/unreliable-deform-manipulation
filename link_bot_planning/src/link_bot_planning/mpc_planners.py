import pathlib
from typing import Tuple, List, Type

import numpy as np
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning import model_utils, classifier_utils, ompl_viz
from link_bot_planning.ompl_viz import VizObject
from link_bot_planning.params import EnvParams, LocalEnvParams, PlannerParams
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

    def plan(self, np_start: np.ndarray,
             tail_goal_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[link_bot_sdf_utils.OccupancyData]]:
        pass


def get_planner(planner_class: Type,
                fwd_model_dir: pathlib.Path,
                fwd_model_type: str,
                classifier_model_dir: pathlib.Path,
                classifier_model_type: str,
                planner_params: PlannerParams,
                local_env_params: LocalEnvParams,
                env_params: EnvParams,
                services: GazeboServices,
                ):
    fwd_model, model_path_info = model_utils.load_generic_model(fwd_model_dir, fwd_model_type)
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, classifier_model_type)
    viz_object = ompl_viz.VizObject()

    planner = planner_class(fwd_model=fwd_model,
                            classifier_model=classifier_model,
                            dt=fwd_model.dt,
                            planner_params=planner_params,
                            local_env_params=local_env_params,
                            env_params=env_params,
                            services=services,
                            viz_object=viz_object,
                            )
    return planner
