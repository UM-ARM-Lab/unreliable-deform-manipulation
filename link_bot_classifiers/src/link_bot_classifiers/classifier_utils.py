import pathlib

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier, DEFAULT_INFLATION_RADIUS
from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_classifiers.rnn_image_classifier import RNNImageClassifierWrapper
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from shape_completion_training.model.filepath_tools import load_trial


def load_generic_model(model_dir: pathlib.Path, scenario: ExperimentScenario) -> [BaseConstraintChecker]:
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param scenario:
    :return:
    """
    if isinstance(model_dir, str):
        model_dir = pathlib.Path(model_dir)
    if isinstance(model_dir, list):
        assert len(model_dir) == 1
        model_dir = model_dir[0]
    _, hparams = load_trial(model_dir.absolute())
    model_type = hparams['model_class']
    if model_type == 'rnn':
        return RNNImageClassifierWrapper(model_dir, batch_size=1, scenario=scenario)
    elif model_type == 'collision':
        return CollisionCheckerClassifier(model_dir, inflation_radius=DEFAULT_INFLATION_RADIUS, scenario=scenario)
    elif model_type == 'none':
        return NoneClassifier(scenario)
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
