from state_space_dynamics import locally_linear_cnn, simple_nn
from state_space_dynamics import locally_linear_nn


def get_model_module(model_class_name):
    if model_class_name == 'LocallyLinearCNN':
        return locally_linear_cnn
    elif model_class_name == 'LocallyLinearNN':
        return locally_linear_nn
    elif  model_class_name == "SimpleNN":
        return simple_nn
