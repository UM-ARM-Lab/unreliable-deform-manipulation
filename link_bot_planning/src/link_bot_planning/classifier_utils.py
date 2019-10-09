import pathlib

from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_classifiers.raster_classifier import RasterClassifier


def load_generic_model(model_dir: pathlib.Path, model_type: str):
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param model_type: string indicating what type of model to load
    :return:
    """
    if model_type == 'raster':
        cnn = RasterClassifier(model_dir)
        return cnn
    elif model_type == 'none':
        # this dt here is sort of made up
        return NoneClassifier()
