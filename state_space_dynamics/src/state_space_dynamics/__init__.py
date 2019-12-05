from state_space_dynamics import simple_nn
from state_space_dynamics import locally_linear_nn
from state_space_dynamics import obstacle_llnn
from state_space_dynamics import obstacle_nn


def get_model_module(model_class_name):
    if model_class_name == 'LocallyLinearNN':
        return locally_linear_nn
    elif model_class_name == "ObstacleLLNN":
        return obstacle_llnn
    elif model_class_name == "ObstacleNN":
        return obstacle_nn
    elif model_class_name == "SimpleNN":
        return simple_nn
