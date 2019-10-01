import ompl.util as ou

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
    if model_type == 'gp':
        fwd_gp_model = link_bot_gp.LinkBotGP(ou.RNG)
        fwd_gp_model.load(model_dir / 'fwd_model')
        return fwd_gp_model
    elif model_type == 'llnn':
        llnn = LocallyLinearNNWrapper(model_dir)
        return llnn
    elif model_type == 'rigid':
        # this dt here is sort of made up
        return RigidTranslationModel(beta=0.7, dt=0.25)
