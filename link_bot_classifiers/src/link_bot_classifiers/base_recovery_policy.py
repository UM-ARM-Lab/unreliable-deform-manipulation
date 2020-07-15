import pathlib
from typing import Dict
import numpy as np

from link_bot_pycommon.experiment_scenario import ExperimentScenario


class BaseRecoveryPolicy:

    def __init__(self, hparams: Dict, model_dir: pathlib.Path, scenario: ExperimentScenario, rng: np.random.RandomState):
        self.hparams = hparams
        self.model_dir = model_dir
        self.scenario = scenario
        self.rng = rng

    def __call__(self, environment: Dict, state: Dict):
        raise NotImplementedError()
