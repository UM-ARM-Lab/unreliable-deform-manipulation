import json
import pathlib

from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier
from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_classifiers.raster_classifier import RasterClassifierWrapper


def load_generic_model(model_dir: pathlib.Path):
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :return:
    """
    model_hparams_file = model_dir / 'hparams.json'
    hparams = json.load(model_hparams_file.open('r'))
    model_type = hparams['model_type']
    if model_type == 'raster':
        return RasterClassifierWrapper(model_dir, batch_size=1)
    elif model_type == 'collision':
        return CollisionCheckerClassifier(inflation_radius=0.02)
    elif model_type == 'none':
        return NoneClassifier()
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
