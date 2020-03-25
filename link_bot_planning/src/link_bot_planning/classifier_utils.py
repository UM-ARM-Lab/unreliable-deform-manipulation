import json
import pathlib

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier
from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_classifiers.raster_classifier import RasterClassifierWrapper
from link_bot_planning.experiment_scenario import ExperimentScenario


def load_generic_model(model_dir: pathlib.Path, scenario: ExperimentScenario) -> [BaseConstraintChecker]:
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param scenario:
    :return:
    """
    model_hparams_file = model_dir / 'hparams.json'
    hparams = json.load(model_hparams_file.open('r'))
    model_type = hparams['model_class']
    if model_type == 'raster':
        return RasterClassifierWrapper(model_dir, batch_size=1, scenario=scenario)
    elif model_type == 'collision':
        return CollisionCheckerClassifier(model_dir, inflation_radius=0.02, scenario=scenario)
    elif model_type == 'none':
        return NoneClassifier(scenario)
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
