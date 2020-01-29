import pathlib

import numpy as np
import ompl.control as oc

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import PlannerParams
from link_bot_planning.state_spaces import TrainingSetCompoundSampler
from link_bot_planning.viz_object import VizObject
from state_space_dynamics.base_forward_model import BaseForwardModel


class BestFirstRRT(MyPlanner):

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
                         viz_object,
                         subspace_weights=[1.0, 100.0, 0.0])

        self.planner = oc.BestFirstRRT(self.si)
        self.planner.setNeighborhoodRadius(planner_params.neighborhood_radius)
        self.planner.setIntermediateStates(True)  # this is necessary!
        self.planner.setGoalBias(0.2)
        self.ss.setPlanner(self.planner)
        self.si.setPropagationStepSize(self.fwd_model.dt)
        self.si.setMinMaxControlDuration(1, 50)

        self.dataset_dirs = [pathlib.Path(p) for p in self.fwd_model.hparams['datasets']]
        dataset = LinkBotStateSpaceDataset(self.dataset_dirs)
        tf_dataset = dataset.get_datasets(mode='train',
                                          shuffle=False,
                                          seed=0,  # doesn't matter, we're not shuffling
                                          sequence_length=None,  # default to max
                                          batch_size=None)  # no batching
        self.training_rope_configurations = []
        for input_data, _ in tf_dataset:
            rope_configurations = input_data['state_s'].numpy().squeeze()
            self.training_rope_configurations.extend(rope_configurations)
        self.training_rope_configurations = np.array(self.training_rope_configurations)

    def state_sampler_allocator(self, state_space):
        sampler = TrainingSetCompoundSampler(state_space,
                                             self.viz_object,
                                             n_state=self.n_state,
                                             rope_configurations=self.training_rope_configurations)
        return sampler
