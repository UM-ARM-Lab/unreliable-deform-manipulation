import pathlib
from typing import List

from link_bot_pycommon.get_scenario import get_scenario
from my_cfm.cfm import CFMFilter
from shape_completion_training.model.filepath_tools import load_trial
from state_space_dynamics.base_filter_function import BaseFilterFunction


def load_filter(model_dirs: List[pathlib.Path]) -> BaseFilterFunction:
    representative_model_dir = model_dirs[0]
    _, common_hparams = load_trial(representative_model_dir.parent.absolute())
    scenario_name = common_hparams['dynamics_dataset_hparams']['scenario']
    scenario = get_scenario(scenario_name)
    model_type = common_hparams['model_class']
    if model_type == 'CFM':
        nn = CFMFilter(model_dirs, batch_size=1, scenario=scenario)
        return nn
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
