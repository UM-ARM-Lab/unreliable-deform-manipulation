import json
import pathlib
from typing import List, Dict

import numpy as np
import rospy
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.collision_checking import batch_in_collision_tf_3d
from link_bot_pycommon.experiment_scenario import ExperimentScenario

DEFAULT_INFLATION_RADIUS = 0.02
# DEFAULT_INFLATION_RADIUS = 0.03
# rospy.logwarn_once("using inflated CC radius")


def check_collision(scenario, environment, states_sequence, collision_check_object=True):
    state = states_sequence[-1]
    if collision_check_object:
        points = scenario.state_to_points_for_cc(state)
    else:
        points = scenario.state_to_gripper_position(state)
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    in_collision, inflated_env = batch_in_collision_tf_3d(environment=environment,
                                                          xs=xs,
                                                          ys=ys,
                                                          zs=zs,
                                                          inflate_radius_m=DEFAULT_INFLATION_RADIUS)
    scenario.plot_environment_rviz({
        'env': inflated_env,
        'res': environment['res'],
        'origin': environment['origin'],
    })
    prediction = tf.expand_dims(tf.logical_not(in_collision), axis=0)
    return prediction


class CollisionCheckerClassifier(BaseConstraintChecker):

    def __init__(self, path: pathlib.Path, inflation_radius: float, scenario: ExperimentScenario):
        super().__init__(scenario)
        self.inflation_radius = inflation_radius
        hparams_file = path.parent / 'params.json'
        self.model_hparams = json.load(hparams_file.open('r'))
        self.local_h_rows = self.model_hparams['local_h_rows']
        self.local_w_cols = self.model_hparams['local_w_cols']
        self.local_c_channels = self.model_hparams['local_c_channels']
        self.horizon = 2
        self.data_collection_params = {
            'res': self.model_hparams['res']
        }

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions) -> tf.Tensor:
        return check_collision(self.scenario, environment, states_sequence)

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: np.ndarray):
        assert len(states_sequence) == 2
        return self.check_constraint_tf(environment, states_sequence, actions).numpy()


model = CollisionCheckerClassifier
