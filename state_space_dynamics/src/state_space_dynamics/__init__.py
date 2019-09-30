from state_space_dynamics.locally_linear_cnn import LocallyLinearCNN
from state_space_dynamics.locally_linear_nn import LocallyLinearNN


def get_model_class(model_class_name):
    if model_class_name == 'LocallyLinearCNN':
        return LocallyLinearCNN
    elif model_class_name == 'LocallyLinearNN':
        return LocallyLinearNN
