from typing import Dict

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.control as oc

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_planning.my_planner import MyPlanner
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class NearestRRT(MyPlanner):

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: BaseConstraintChecker,
                 planner_params: Dict,
                 scenario: ExperimentScenario,
                 verbose: int):
        super().__init__(fwd_model,
                         classifier_model,
                         planner_params,
                         scenario,
                         verbose)

        self.rrt = oc.RRT(self.si)
        self.rrt.setIntermediateStates(True)  # this is necessary, because we use this to generate datasets
        self.ss.setPlanner(self.rrt)
        self.si.setMinMaxControlDuration(1, 50)
