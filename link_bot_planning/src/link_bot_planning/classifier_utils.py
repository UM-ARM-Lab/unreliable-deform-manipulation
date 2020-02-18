import pathlib

from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier
from link_bot_classifiers.feature_classifier import FeatureClassifierWrapper
from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_classifiers.raster_classifier import RasterClassifierWrapper
from link_bot_classifiers.hand_designed_obs_classifier import HandDesignedObsClassifier
from link_bot_classifiers.human_classifier import ManualClassifier


def load_generic_model(model_dir: pathlib.Path, model_type: str):
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param model_type: string indicating what type of model to load
    :return:
    """
    if model_type == 'raster':
        return RasterClassifierWrapper(model_dir, batch_size=1)
    elif model_type == 'collision':
        return CollisionCheckerClassifier(inflation_radius=0.02)
    elif model_type == 'none':
        return NoneClassifier()
    elif model_type == 'feature':
        return FeatureClassifierWrapper(model_dir, batch_size=1)
    elif model_type == 'designed':
        return HandDesignedObsClassifier(model_dir, batch_size=1)
    elif model_type == 'manual':
        return ManualClassifier()
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
