import json
import pathlib

from link_bot_classifiers.rnn_image_classifier import RNNImageClassifierWrapper
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier
from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_classifiers.single_image_classifier import SingleImageClassifierWrapper
from link_bot_pycommon.experiment_scenario import ExperimentScenario


def load_generic_model(model_dir: pathlib.Path, scenario: ExperimentScenario) -> [BaseConstraintChecker]:
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param scenario:
    :return:
    """
    if isinstance(model_dir, str):
        model_dir = pathlib.Path(model_dir)
    model_hparams_file = model_dir / 'hparams.json'
    hparams = json.load(model_hparams_file.open('r'))
    model_type = hparams['model_class']
    if model_type == 'raster':
        return SingleImageClassifierWrapper(model_dir, batch_size=1, scenario=scenario)
    elif model_type == 'rnn':
        return RNNImageClassifierWrapper(model_dir, batch_size=1, scenario=scenario)
    elif model_type == 'collision':
        return CollisionCheckerClassifier(model_dir, inflation_radius=0.02, scenario=scenario)
    elif model_type == 'none':
        return NoneClassifier(scenario)
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
