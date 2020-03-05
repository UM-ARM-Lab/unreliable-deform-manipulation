from typing import List, Dict

import numpy as np

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.link_bot_pycommon import vector_to_points_2d
from link_bot_pycommon.link_bot_sdf_utils import point_to_idx, OccupancyData, get_local_env_and_origin


class CollisionCheckerClassifier(BaseConstraintChecker):

    def __init__(self, inflation_radius: float, scenario: ExperimentScenario):
        super().__init__(scenario)
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

    def check_transition(self,
                         local_env: np.ndarray,
                         local_env_origin: np.ndarray,
                         res: float,
                         s1: np.ndarray,
                         s2: np.ndarray,
                         action: np.ndarray) -> float:
        del action  # unused
        local_env_data = OccupancyData(data=local_env, origin=local_env_origin, resolution=res)
        inflated_local_env = link_bot_sdf_utils.inflate(local_env_data, self.inflation_radius)
        xs1, ys1 = vector_to_points_2d(s1)
        xs2, ys2 = vector_to_points_2d(s2)

        first_point_check = self.check_collision(inflated_local_env.data, res, local_env_origin, xs1, ys1)
        second_point_check = self.check_collision(inflated_local_env.data, res, local_env_origin, xs2, ys2)
        prediction = 1.0 if (first_point_check and second_point_check) else 0.0
        return prediction

    def check_trajectory(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        raise NotImplementedError()

    def check_constraint(self,
                         full_env: np.ndarray,
                         full_env_origin: np.ndarray,
                         res: float,
                         states_sequence: List[Dict],
                         actions: np.ndarray) -> float:
        states_i = states_sequence[-2]
        # TODO: put h_rows/w_cols in hparams file?
        local_env_center = self.get_local_environment_center(states_i)
        local_env, local_env_origin = get_local_env_and_origin(center_point=local_env_center,
                                                               full_env=full_env,
                                                               full_env_origin=full_env_origin,
                                                               res=res,
                                                               local_h_rows=50,
                                                               local_w_cols=50)
        # remove batch dim with [0]
        return self.check_transition(local_env=local_env,
                                     local_env_origin=local_env_origin,
                                     res=res,
                                     s1=states_sequence[-2]['link_bot'],
                                     s2=states_sequence[-1]['link_bot'],
                                     action=actions[-1])


model = CollisionCheckerClassifier
