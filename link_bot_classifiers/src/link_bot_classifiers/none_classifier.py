import numpy as np

from link_bot_planning.my_motion_validator import MotionClassifier
from link_bot_pycommon import link_bot_sdf_utils


class NoneClassifier(MotionClassifier):

    def __init__(self):
        super().__init__()

    def predict(self, local_sdf_data: link_bot_sdf_utils.SDF, s1: np.ndarray, s2: np.ndarray):
        return 1.0
