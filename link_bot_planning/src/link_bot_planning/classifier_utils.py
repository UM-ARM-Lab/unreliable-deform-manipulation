import pathlib

from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier
from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_classifiers.raster_classifier import RasterClassifierWrapper
from link_bot_classifiers.ensemble_classifier import EnsembleClassifier


def load_generic_model(model_dir: pathlib.Path, model_type: str):
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param model_type: string indicating what type of model to load
    :return:
    """
    if model_type == 'raster':
        return RasterClassifierWrapper(model_dir)
    elif model_type == 'collision':
        return CollisionCheckerClassifier(inflation_radius=0.02)
    elif model_type == 'none':
        return NoneClassifier()
    elif model_type == 'ensemble':
        return EnsembleClassifier()
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
