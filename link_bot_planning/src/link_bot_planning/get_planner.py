import pathlib
from typing import Dict

from link_bot_classifiers import classifier_utils
from state_space_dynamics import model_utils
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_planning.nearest_rrt import NearestRRT
from link_bot_pycommon.base_services import BaseServices
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def get_planner(planner_params: Dict, verbose: int):
    fwd_model_dirs = [pathlib.Path(model_dir) for model_dir in planner_params['fwd_model_dir']]

    fwd_model, model_path_info = model_utils.load_generic_model(fwd_model_dirs)
    scenario = fwd_model.scenario

    classifier_model_dir = pathlib.Path(planner_params['classifier_model_dir'])
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, scenario=scenario)

    planner = NearestRRT(fwd_model=fwd_model,
                         classifier_model=classifier_model,
                         planner_params=planner_params,
                         scenario=scenario,
                         verbose=verbose)
    return planner, model_path_info


def get_planner_with_model(planner_class_str: str,
                           fwd_model: BaseDynamicsFunction,
                           classifier_model_dir: pathlib.Path,
                           planner_params: Dict,
                           verbose: int):
    scenario = get_scenario(planner_params['scenario'])
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, scenario)

    if planner_class_str == 'NearestRRT':
        planner_class = NearestRRT
    else:
        raise ValueError(planner_class_str)

    planner = planner_class(fwd_model=fwd_model,
                            classifier_model=classifier_model,
                            planner_params=planner_params,
                            scenario=scenario,
                            verbose=verbose,
                            )
    return planner
