from ompl import control as oc

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import SimParams, PlannerParams
from link_bot_planning.viz_object import VizObject
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class SST(MyPlanner):

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: BaseConstraintChecker,
                 planner_params: PlannerParams,
                 services: GazeboServices,
                 viz_object: VizObject,
                 seed: int):
        super().__init__(fwd_model,
                         classifier_model,
                         planner_params,
                         services,
                         viz_object,
                         seed)

        self.planner = oc.SST(self.si)
        self.ss.setPlanner(self.planner)
        self.si.setPropagationStepSize(self.fwd_model.dt)
        self.si.setMinMaxControlDuration(1, 50)
