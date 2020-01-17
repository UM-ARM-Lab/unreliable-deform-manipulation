from ompl import control as oc

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import EnvParams, PlannerParams
from link_bot_planning.viz_object import VizObject
from state_space_dynamics.base_forward_model import BaseForwardModel


class SST(MyPlanner):

    def __init__(self,
                 fwd_model: BaseForwardModel,
                 classifier_model: BaseClassifier,
                 planner_params: PlannerParams,
                 env_params: EnvParams,
                 services: GazeboServices,
                 viz_object: VizObject):
        super().__init__(fwd_model,
                         classifier_model,
                         planner_params,
                         env_params,
                         services,
                         viz_object)

        self.planner = oc.SST(self.si)
        self.ss.setPlanner(self.planner)
        self.si.setPropagationStepSize(self.fwd_model.dt)
        self.si.setMinMaxControlDuration(1, 50)
