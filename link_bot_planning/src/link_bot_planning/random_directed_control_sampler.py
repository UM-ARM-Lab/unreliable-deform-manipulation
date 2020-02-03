import numpy as np
import ompl.control as oc


class RandomDirectedControlSampler(oc.DirectedControlSampler):

    def __init__(self, si):
        super().__init__(si)
        self.control_space = self.si.getControlSpace()

    @classmethod
    def alloc(cls, si):
        return cls(si)

    @classmethod
    def allocator(cls):
        return oc.DirectedControlSamplerAllocator(cls.alloc)

    def sampleTo(self, control_out, previous_control, state, target_out):
        # unused
        del state
        del target_out
        del previous_control
        min_step_count = self.si.getMinControlDuration()
        max_step_count = self.si.getMaxControlDuration()
        random_control = self.control_space.sample()
        print(random_control, min_step_count, max_step_count)
        self.control_space.from_numpy(random_control, control_out)
