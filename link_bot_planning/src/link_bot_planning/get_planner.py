import pathlib
from typing import Dict

import link_bot_planning.viz_object
from link_bot_planning import model_utils, classifier_utils
from link_bot_planning.get_scenario import get_scenario
from link_bot_planning.nearest_rrt import NearestRRT
from link_bot_pycommon.base_services import Services
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.ensemble_dynamics_function import EnsembleDynamicsFunction


def get_planner(planner_params: Dict, service_provider: Services, seed: int, verbose: int):
    fwd_model_dirs = planner_params['fwd_model_dir']
    if isinstance(fwd_model_dirs, list):
        fwd_model_dirs = [pathlib.Path(d) for d in fwd_model_dirs]
        fwd_model = EnsembleDynamicsFunction(fwd_model_dirs, batch_size=1)
        scenario = fwd_model.scenario
        model_path_info = list(fwd_model_dirs[0].parts[1:])
        model_path_info[-1] = model_path_info[-1][:-2]  # remove the "-$n" so it's "dir/ensemble" instead of "dir/ensemble-$n"
    else:
        fwd_model, model_path_info = model_utils.load_generic_model(pathlib.Path(fwd_model_dirs))
        scenario = fwd_model.scenario

    classifier_model_dir = pathlib.Path(planner_params['classifier_model_dir'])
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, scenario=scenario)
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
                            service_provider=service_provider,
                            scenario=scenario,
                            seed=seed,
                            verbose=verbose,
                            )
    return planner, model_path_info


def get_planner_with_model(planner_class_str: str,
                           fwd_model: BaseDynamicsFunction,
                           classifier_model_dir: pathlib.Path,
                           planner_params: Dict,
                           service_provider: Services,
                           seed: int,
                           verbose: int):
    scenario = get_scenario(planner_params['scenario'])
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, scenario)
    viz_object = link_bot_planning.viz_object.VizObject()

    if planner_class_str == 'NearestRRT':
        planner_class = NearestRRT
    else:
        raise ValueError(planner_class_str)

    planner = planner_class(fwd_model=fwd_model,
                            classifier_model=classifier_model,
                            planner_params=planner_params,
                            viz_object=viz_object,
                            scenario=scenario,
                            service_provider=service_provider,
                            seed=seed,
                            verbose=verbose,
                            )
    return planner
