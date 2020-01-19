import ompl.control as oc

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import PlannerParams
from link_bot_planning.viz_object import VizObject
from state_space_dynamics.base_forward_model import BaseForwardModel


class ShootingRRT(MyPlanner):

    def __init__(self,
                 fwd_model: BaseForwardModel,
                 classifier_model: BaseClassifier,
                 planner_params: PlannerParams,
                 services: GazeboServices,
                 viz_object: VizObject):
        super().__init__(fwd_model,
                         classifier_model,
                         planner_params,
                         services,
                         viz_object)

        self.planner = oc.RRT(self.si)
        self.planner.setIntermediateStates(True)  # this is necessary!
        self.ss.setPlanner(self.planner)
        self.si.setPropagationStepSize(self.fwd_model.dt)
        self.si.setMinMaxControlDuration(1, 50)

        # TODO: make a parameter for k
        # dcs_allocator = oc.DirectedControlSamplerAllocator(lambda si: oc.SimpleDirectedControlSampler(si, k=10))
        # self.si.setDirectedControlSamplerAllocator(dcs_allocator)
