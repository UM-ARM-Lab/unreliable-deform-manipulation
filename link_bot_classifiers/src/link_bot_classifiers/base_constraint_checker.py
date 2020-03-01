from typing import Dict

import numpy as np

from link_bot_pycommon.link_bot_sdf_utils import OccupancyData


class BaseConstraintChecker:

    def __init__(self):
        self.model_hparams = {}
        self.full_env_params = None

    def check_constraint(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        pass

    def check_transition(self, local_env_data: OccupancyData, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> float:
        pass

    def check_traj(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        pass
