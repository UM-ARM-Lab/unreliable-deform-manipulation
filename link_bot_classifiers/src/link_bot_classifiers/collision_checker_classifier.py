from typing import List

import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_data.visualization import plottable_rope_configuration
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.link_bot_sdf_utils import point_to_idx, OccupancyData


class CollisionCheckerClassifier(BaseClassifier):

    def __init__(self, inflation_radius: float):
        super().__init__()
        self.inflation_radius = inflation_radius

    def predict_state_only(self, local_env_data_s: List[OccupancyData], s1_s: np.ndarray) -> float:
        predictions = []
        for local_env, s1, s2 in zip(local_env_data_s, s1_s):
            local_env = link_bot_sdf_utils.inflate(local_env, self.inflation_radius)
            xs, ys = plottable_rope_configuration(s1)

            prediction = True
            for x, y in zip(xs, ys):
                row, col = point_to_idx(x, y, local_env.resolution, origin=local_env.origin)
                try:
                    # 1 means obstacle, aka in collision
                    d = local_env.data[row, col]
                    point_not_in_collision = not d
                    # prediction of True means not in collision
                    prediction = prediction and point_not_in_collision
                except IndexError:
                    pass

            prediction = 1.0 if prediction else 0.0
            predictions.append(prediction)
        return predictions

    def predict(self, local_env_data: List[OccupancyData], s1_s: np.ndarray, s2_s: np.ndarray) -> float:
        predictions = []
        for local_env, s1, s2 in zip(local_env_data, s1_s, s2_s):
            # local_env = link_bot_sdf_utils.inflate(local_env, self.inflation_radius)
            xs1, ys1 = plottable_rope_configuration(s1)
            xs2, ys2 = plottable_rope_configuration(s2)

            def _check(xs, ys):
                prediction = True
                for x, y in zip(xs, ys):
                    row, col = point_to_idx(x, y, local_env.resolution, origin=local_env.origin)
                    try:
                        # 1 means obstacle, aka in collision
                        d = local_env.data[row, col]
                        point_not_in_collision = not d
                        # prediction of True means not in collision
                        prediction = prediction and point_not_in_collision
                    except IndexError:
                        pass
                return prediction

            prediction = _check(xs1, ys1)
            prediction = prediction and _check(xs2, ys2)
            prediction = 1.0 if prediction else 0.0
            predictions.append(prediction)

        return predictions
