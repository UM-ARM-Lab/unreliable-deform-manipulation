import json
import pathlib
from typing import List, Tuple

import ompl.util as ou

from link_bot_gaussian_process import link_bot_gp
from state_space_dynamics.locally_linear_nn import LocallyLinearNNWrapper
from state_space_dynamics.rigid_translation_model import RigidTranslationModel
from state_space_dynamics.simple_nn import SimpleNNWrapper


def load_generic_model(model_dir: pathlib.Path, model_type: str) -> [object, Tuple[str]]:
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param model_type: string indicating what type of model to load
    :return: the model class, and a list of strings describing the model
    """
    if model_type == 'gp':
        fwd_gp_model = link_bot_gp.LinkBotGP(ou.RNG)
        fwd_gp_model.load(model_dir / 'fwd_model')
        return fwd_gp_model, model_dir.parts[1:]
    elif model_type == 'llnn':
        llnn = LocallyLinearNNWrapper(model_dir)
        return llnn, model_dir.parts[1:]
    elif model_type == 'rigid':
        # this dt here is sort of made up
        hparams = json.load((model_dir / 'hparams.json').open('r'))
        return RigidTranslationModel(beta=hparams['beta'], dt=hparams['dt']), model_dir.parts[1:]
    elif model_type == 'nn':
        nn = SimpleNNWrapper(model_dir)
        return nn, model_dir.parts[1:]


def get_model_info(model_dir: pathlib.Path, model_type: str) -> Tuple[str]:
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param model_type: string indicating what type of model to load
    :return: the model class, and a list of strings describing the model
    """
    if model_type == 'gp':
        return model_dir.parts[1:]
    elif model_type == 'llnn':
        return model_dir.parts[1:]
    elif model_type == 'rigid':
        # this dt here is sort of made up
        return model_dir.parts[1:]
    elif model_type == 'nn':
        return model_dir.parts[1:]
