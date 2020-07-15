import pathlib
import json

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_classifiers.random_recovery_policy import RandomRecoveryPolicy
from link_bot_classifiers.simple_recovery_policy import SimpleRecoveryPolicy


def load_generic_model(model_dir: pathlib.Path, scenario: ExperimentScenario):
    with (model_dir / 'hparams.json').open('r') as hparams_file:
        hparams = json.load(hparams_file)

    model_type = hparams['type']
    if model_type == 'simple':
        return SimpleRecoveryPolicy(hparams, model_dir, scenario)
    elif model_type == 'random':
        return RandomRecoveryPolicy(hparams, model_dir, scenario)
    else:
        raise NotImplementedError(f"model type {model_type} not implemented")
