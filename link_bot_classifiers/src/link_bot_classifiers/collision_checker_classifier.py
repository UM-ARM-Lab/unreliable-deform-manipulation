import json
import pathlib
from typing import List, Dict
import tensorflow as tf

import numpy as np

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.link_bot_pycommon import vector_to_points_2d
from link_bot_pycommon.link_bot_sdf_utils import point_to_idx, OccupancyData
from moonshine.get_local_environment import get_local_env_and_origin


class CollisionCheckerClassifier(BaseConstraintChecker):

    def __init__(self, path: pathlib.Path, inflation_radius: float, scenario: ExperimentScenario):
        super().__init__(scenario)
        self.inflation_radius = inflation_radius
        model_hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.local_h_rows = self.model_hparams['local_h_rows']
        self.local_w_cols = self.model_hparams['local_w_cols']
        self.horizon = 2

    @staticmethod
    def check_collision(inflated_local_env, resolution, origin, xs, ys):
        prediction = True
        for x, y in zip(xs, ys):
            row, col = point_to_idx(x, y, resolution, origin=origin)
            try:
                # 1 means obstacle, aka in collision
                d = inflated_local_env[row, col] > 0.5
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
        xs1, ys1 = vector_to_points_2d(s1)
        xs2, ys2 = vector_to_points_2d(s2)

        inflated_local_env = link_bot_sdf_utils.inflate(local_env_data, self.inflation_radius)
        first_point_check = self.check_collision(inflated_local_env.data, res, local_env_origin, xs1, ys1)
        second_point_check = self.check_collision(inflated_local_env.data, res, local_env_origin, xs2, ys2)
        prediction = 1.0 if (first_point_check and second_point_check) else 0.0

        return prediction

    def check_trajectory(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray):
        raise NotImplementedError()

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: np.ndarray):
        states_i = states_sequence[-1]
        full_env = environment['full_env/env']
        full_env_origin = environment['full_env/origin']
        res = environment['full_env/res']
        local_env_center = tf.cast(self.scenario.local_environment_center(states_i), tf.float32)

        local_env, local_env_origin = get_local_env_and_origin(center_point=local_env_center,
                                                               full_env=full_env,
                                                               full_env_origin=full_env_origin,
                                                               res=res,
                                                               local_h_rows=self.local_h_rows,
                                                               local_w_cols=self.local_w_cols)
        # remove batch dim with [0]
        # TODO: use scenario here instead of hard-coding link_bot?
        prediction = self.check_transition(local_env=local_env,
                                     local_env_origin=local_env_origin,
                                     res=res,
                                     s1=states_sequence[-2]['link_bot'],
                                     s2=states_sequence[-1]['link_bot'],
                                     action=actions[-1])
        return [prediction]


model = CollisionCheckerClassifier
