from typing import Dict, List

from link_bot_pycommon.experiment_scenario import ExperimentScenario


class BaseRecoveryActionsModels:

    def __init__(self, scenario: ExperimentScenario):
        self.scenario = scenario
        self.model_hparams = {}
        self.full_env_params = None

    def sample(self,
               environment: Dict,
               state: Dict,
               ):
        raise NotImplementedError()
