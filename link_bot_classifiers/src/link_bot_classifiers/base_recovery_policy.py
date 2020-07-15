import pathlib
from typing import Dict

from link_bot_pycommon.experiment_scenario import ExperimentScenario


class BaseRecoveryPolicy:

    def __init__(self, hparams: Dict, model_dir: pathlib.Path, scenario: ExperimentScenario):
        self.hparams = hparams
        self.model_dir = model_dir
        self.scenario = scenario

    def __call__(self, environment: Dict, state: Dict):
        raise NotImplementedError()
