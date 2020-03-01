from typing import List, Dict

import numpy as np

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_data.visualization import plottable_rope_configuration
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.link_bot_sdf_utils import point_to_idx, OccupancyData
from moonshine.numpy_utils import add_batch


class CollisionCheckerClassifier(BaseConstraintChecker):

    def __init__(self, inflation_radius: float):
        super().__init__()
        self.inflation_radius = inflation_radius

    @staticmethod
    def check_collision(inflated_local_env, resolution, origin, xs, ys):
        prediction = True
        for x, y in zip(xs, ys):
            row, col = point_to_idx(x, y, resolution, origin=origin)
            try:
                # 1 means obstacle, aka in collision
                d = inflated_local_env[row, col]
            except IndexError:
                # assume out of bounds is free space
                continue
            point_not_in_collision = not d
            # prediction of True means not in collision
            prediction = prediction and point_not_in_collision
        return prediction

    def check_transition(self, local_env_data: OccupancyData, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> float:
        inflated_local_env = link_bot_sdf_utils.inflate(local_env_data, self.inflation_radius)
        xs1, ys1 = plottable_rope_configuration(s1)
        xs2, ys2 = plottable_rope_configuration(s2)

        first_point_check = self.check_collision(inflated_local_env.data, local_env_data.resolution, local_env_data.origin, xs1,
                                                 ys1)
        second_point_check = self.check_collision(inflated_local_env.data, local_env_data.resolution, local_env_data.origin, xs2,
                                                  ys2)
        prediction = 1.0 if (first_point_check and second_point_check) else 0.0
        return prediction

    def check_traj(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        raise NotImplementedError()

    def check_constraint(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        head_point = states['link_bot'][-2]
        local_env, local_env_origin = link_bot_sdf_utils.get_local_env_and_origin(*add_batch(head_point,
                                                                                             full_env.data,
                                                                                             full_env.origin),
                                                                                  local_h_rows=50,
                                                                                  local_w_cols=50,
                                                                                  res=full_env.resolution[0])
        # remove batch dim with [0]
        local_env_data = OccupancyData(data=local_env[0],
                                       resolution=full_env.resolution,
                                       origin=local_env_origin[0])
        return self.check_transition(local_env_data=local_env_data,
                                     s1=states['link_bot'][-2],
                                     s2=states['link_bot'][-1],
                                     action=actions[-1])


model = CollisionCheckerClassifier
