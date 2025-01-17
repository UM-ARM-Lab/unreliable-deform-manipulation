import json
import pathlib
from typing import List, Dict, Optional

import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.collision_checking import batch_in_collision_tf_3d
from link_bot_pycommon.experiment_scenario import ExperimentScenario

DEFAULT_INFLATION_RADIUS = 0.02


def check_collision(scenario, environment, states_sequence, collision_check_object=True):
    state = states_sequence[-1]
    if collision_check_object:
        points = scenario.state_to_points_for_cc(state)
    else:
        points = scenario.state_to_gripper_position(state)
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    in_collision, _ = batch_in_collision_tf_3d(environment=environment,
                                               xs=xs,
                                               ys=ys,
                                               zs=zs,
                                               inflate_radius_m=DEFAULT_INFLATION_RADIUS)
    prediction = tf.expand_dims(tf.logical_not(in_collision), axis=0)
    return prediction


class CollisionCheckerClassifier(BaseConstraintChecker):

    def __init__(self,
                 paths: List[pathlib.Path],
                 scenario: ExperimentScenario,
                 inflation_radius: Optional[float] = DEFAULT_INFLATION_RADIUS,
                 ):
        super().__init__(paths, scenario)
        assert len(paths) == 1
        self.path = paths[0]
        self.inflation_radius = inflation_radius
        hparams_file = self.path.parent / 'params.json'
        self.hparams = json.load(hparams_file.open('r'))
        self.local_h_rows = self.hparams['local_h_rows']
        self.local_w_cols = self.hparams['local_w_cols']
        self.local_c_channels = self.hparams['local_c_channels']
        self.horizon = 2
        self.data_collection_params = {
            'res': self.hparams['res']
        }

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions):
        return check_collision(self.scenario, environment, states_sequence), tf.ones([], dtype=tf.float32) * 1e-9

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        assert len(states_sequence) == 2
        c, s = self.check_constraint_tf(environment, states_sequence, actions)
        return c.numpy, s.numpy()


model = CollisionCheckerClassifier
