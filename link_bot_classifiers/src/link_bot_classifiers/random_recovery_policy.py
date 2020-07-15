from typing import Dict
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy


class RandomRecoveryPolicy(BaseRecoveryPolicy):

    def __init__(self, scenario: ExperimentScenario):
        self.scenario = scenario

    def __call__(self, environment: Dict, state: Dict):
        self.scenario.sample_action(environment=environment,
                                    state=state)
