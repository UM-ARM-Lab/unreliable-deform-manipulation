from typing import Dict, List

from link_bot_pycommon.experiment_scenario import ExperimentScenario


class BaseConstraintChecker:

    def __init__(self, scenario: ExperimentScenario):
        self.scenario = scenario
        self.model_hparams = {}

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        raise NotImplementedError()

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: List[Dict]):
        raise NotImplementedError()
