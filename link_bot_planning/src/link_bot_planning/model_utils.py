import pathlib
from typing import Tuple

from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.obstacle_nn import ObstacleNNWrapper
from state_space_dynamics.rigid_translation_model import RigidTranslationModel
from state_space_dynamics.simple_nn import SimpleNNWrapper


def load_generic_model(model_dir: pathlib.Path, model_type: str) -> [BaseDynamicsFunction, Tuple[str]]:
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param model_type: string indicating what type of model to load
    :return: the model class, and a list of strings describing the model
    """
    if model_type == 'rigid':
        # this dt here is sort of made up
        return RigidTranslationModel(model_dir), model_dir.parts[1:]
    elif model_type == 'nn':
        nn = SimpleNNWrapper(model_dir)
        return nn, model_dir.parts[1:]
    elif model_type == 'obs':
        nn = ObstacleNNWrapper(model_dir, batch_size=1)
        return nn, model_dir.parts[1:]
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))


def get_model_info(model_dir: pathlib.Path, model_type: str) -> Tuple[str]:
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param model_type: string indicating what type of model to load
    :return: the model class, and a list of strings describing the model
    """
    if model_type == 'rigid':
        return model_dir.parts[1:]
    elif model_type == 'nn':
        return model_dir.parts[1:]
    elif model_type == 'obs':
        return model_dir.parts[1:]
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))
