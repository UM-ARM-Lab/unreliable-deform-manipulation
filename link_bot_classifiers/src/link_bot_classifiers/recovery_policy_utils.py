import pathlib
import numpy as np
import json

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_classifiers.random_recovery_policy import RandomRecoveryPolicy
from link_bot_classifiers.simple_recovery_policy import SimpleRecoveryPolicy
from link_bot_classifiers.nn_recovery_policy import NNRecoveryPolicy
from link_bot_classifiers.knn_recovery_policy import KNNRecoveryPolicy


def load_generic_model(model_dir: pathlib.Path, scenario: ExperimentScenario, rng: np.random.RandomState):
    with (model_dir / 'params.json').open('r') as hparams_file:
        hparams = json.load(hparams_file)

    model_class = hparams['model_class']
    if model_class == 'simple':
        return SimpleRecoveryPolicy(hparams, model_dir, scenario, rng)
    elif model_class == 'random':
        return RandomRecoveryPolicy(hparams, model_dir, scenario, rng)
    elif model_class == 'nn':
        return NNRecoveryPolicy(hparams, model_dir, scenario, rng)
    elif model_class == 'knn':
        return KNNRecoveryPolicy(hparams, model_dir, scenario, rng)
    else:
        raise NotImplementedError(f"model type {model_class} not implemented")
