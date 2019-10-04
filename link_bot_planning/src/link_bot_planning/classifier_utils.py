import ompl.util as ou

from link_bot_classifiers.none_classifier import NoneClassifier
from link_bot_gaussian_process import link_bot_gp
from state_space_dynamics.locally_linear_nn import LocallyLinearNNWrapper
from state_space_dynamics.rigid_translation_model import RigidTranslationModel


def load_generic_model(model_dir, model_type):
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :param model_type: string indicating what type of model to load
    :return:
    """
    if model_type == 'cnn':
        cnn = SimpleCNNWrapper(model_dir)
        return cnn
    elif model_type == 'none':
        # this dt here is sort of made up
        return NoneClassifier()
