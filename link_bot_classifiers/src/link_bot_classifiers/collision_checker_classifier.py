from typing import List

import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier
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
            tail_x = s1[:, 0]
            tail_y = s1[:, 1]
            mid_x = s1[:, 2]
            mid_y = s1[:, 3]
            head_x = s1[:, 4]
            head_y = s1[:, 5]

            tail_row, tail_col = point_to_idx(tail_x, tail_y, local_env.resolution, origin=local_env.origin)
            mid_row, mid_col = point_to_idx(mid_x, mid_y, local_env.resolution, origin=local_env.origin)
            head_row, head_col = point_to_idx(head_x, head_y, local_env.resolution, origin=local_env.origin)

            try:
                tail_d = local_env.data[tail_row, tail_col]
            except IndexError:
                tail_d = np.inf
            try:
                mid_d = local_env.data[mid_row, mid_col]
            except IndexError:
                mid_d = np.inf
            head_d = local_env.data[head_row, head_col]

            # prediction == 1 means not in collision
            prediction = not (tail_d or mid_d or head_d)
            prediction = 1.0 if prediction else 0.0
            predictions.append(prediction)
        return predictions

    def predict(self, local_env_data: List[OccupancyData], s1_s: np.ndarray, s2_s: np.ndarray) -> float:
        predictions = []
        for local_env, s1, s2 in zip(local_env_data, s1_s, s2_s):
            tail_x = s1[0]
            tail_y = s1[1]
            mid_x = s1[2]
            mid_y = s1[3]
            head_x = s1[4]
            head_y = s1[5]
            next_tail_x = s2[0]
            next_tail_y = s2[1]
            next_mid_x = s2[2]
            next_mid_y = s2[3]
            next_head_x = s2[4]
            next_head_y = s2[5]

            local_env = link_bot_sdf_utils.inflate(local_env, self.inflation_radius)

            tail_row, tail_col = point_to_idx(tail_x, tail_y, local_env.resolution, origin=local_env.origin)
            mid_row, mid_col = point_to_idx(mid_x, mid_y, local_env.resolution, origin=local_env.origin)
            head_row, head_col = point_to_idx(head_x, head_y, local_env.resolution, origin=local_env.origin)
            next_tail_row, next_tail_col = point_to_idx(next_tail_x, next_tail_y, local_env.resolution, origin=local_env.origin)
            next_mid_row, next_mid_col = point_to_idx(next_mid_x, next_mid_y, local_env.resolution, origin=local_env.origin)
            next_head_row, next_head_col = point_to_idx(next_head_x, next_head_y, local_env.resolution, origin=local_env.origin)

            try:
                tail_d = local_env.data[tail_row, tail_col]
            except IndexError:
                tail_d = np.inf
            try:
                next_tail_d = local_env.data[next_tail_row, next_tail_col]
            except IndexError:
                next_tail_d = np.inf
            try:
                mid_d = local_env.data[mid_row, mid_col]
            except IndexError:
                mid_d = np.inf
            try:
                next_mid_d = local_env.data[next_mid_row, next_mid_col]
            except IndexError:
                next_mid_d = np.inf
            head_d = local_env.data[head_row, head_col]
            next_head_d = local_env.data[next_head_row, next_head_col]

            # prediction == 1 means not in collision
            prediction = not (tail_d or mid_d or head_d or next_tail_d or next_mid_d or next_head_d)
            prediction = 1.0 if prediction else 0.0
            predictions.append(prediction)
        return predictions
