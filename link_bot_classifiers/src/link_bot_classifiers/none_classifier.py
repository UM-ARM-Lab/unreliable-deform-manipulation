from typing import Dict

import numpy as np

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.link_bot_sdf_utils import OccupancyData


class NoneClassifier(BaseConstraintChecker):

    def __init__(self):
        super().__init__()

    def check_transition(self, local_env_data: OccupancyData, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> float:
        return 1.0

    def check_traj(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        return 1.0

    def check_constraint(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        return 1.0


model = NoneClassifier
