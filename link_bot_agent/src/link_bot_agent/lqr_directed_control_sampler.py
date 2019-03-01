import numpy as np
import ompl.util as ou
from link_bot_agent.my_directed_control_sampler import MyDirectedControlSampler

class LQRDirectedControlSampler(MyDirectedControlSampler):

    def __init__(self, si):
        super(LQRDirectedControlSampler, self).__init__(si, "LQR")

    def sampleTo(self, sampler, control, state, target):
        duration_steps = 1
        return duration_steps

