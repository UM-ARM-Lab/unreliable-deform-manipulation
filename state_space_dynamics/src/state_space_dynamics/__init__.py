from state_space_dynamics import simple_nn
from state_space_dynamics import obstacle_nn


def get_model(model_class_name):
    if model_class_name == "ObstacleNN":
        return obstacle_nn.ObstacleNN
    elif model_class_name == "SimpleNN":
        return simple_nn.SimpleNN

