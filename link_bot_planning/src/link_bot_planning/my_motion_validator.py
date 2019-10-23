import numpy as np

from link_bot_pycommon import link_bot_sdf_utils


class MotionClassifier:

    def __init__(self):
        self.model_hparams = {}

    def predict(self, local_env_data: link_bot_sdf_utils.OccupancyData, s1: np.ndarray, s2: np.ndarray) -> float:
        pass
