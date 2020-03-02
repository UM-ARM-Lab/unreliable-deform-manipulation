import pathlib
from typing import Dict

import link_bot_planning.viz_object
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning import model_utils, classifier_utils
from link_bot_planning.nearest_rrt import NearestRRT
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def get_planner(planner_params: Dict, services: GazeboServices, seed: int):
    fwd_model_dir = pathlib.Path(planner_params['fwd_model_dir'])
    classifier_model_dir = pathlib.Path(planner_params['classifier_model_dir'])
    fwd_model, model_path_info = model_utils.load_generic_model(fwd_model_dir, planner_params['fwd_model_type'])
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, planner_params['classifier_model_type'])
    viz_object = link_bot_planning.viz_object.VizObject()

    planner_class_str = planner_params['planner_type']
    if planner_class_str == 'NearestRRT':
        planner_class = NearestRRT
    else:
        raise ValueError(planner_class_str)

    planner = planner_class(fwd_model=fwd_model,
                            classifier_model=classifier_model,
                            planner_params=planner_params,
                            viz_object=viz_object,
                            services=services,
                            seed=seed,
                            )
    return planner, model_path_info


def get_planner_with_model(planner_class_str: str,
                           fwd_model: BaseDynamicsFunction,
                           classifier_model_dir: pathlib.Path,
                           classifier_model_type: str,
                           planner_params: Dict,
                           services: GazeboServices,
                           seed: int):
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, classifier_model_type)
    viz_object = link_bot_planning.viz_object.VizObject()

    if planner_class_str == 'NearestRRT':
        planner_class = NearestRRT
    else:
        raise ValueError(planner_class_str)

    planner = planner_class(fwd_model=fwd_model,
                            classifier_model=classifier_model,
                            planner_params=planner_params,
                            viz_object=viz_object,
                            services=services,
                            seed=seed,
                            )
    return planner
