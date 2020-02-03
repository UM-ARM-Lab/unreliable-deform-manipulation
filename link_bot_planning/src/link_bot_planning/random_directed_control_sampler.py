import numpy as np
import ompl.control as oc


class RandomDirectedControlSampler(oc.DirectedControlSampler):

    def __init__(self, si):
        super().__init__(si)
        self.si = si
        self.control_space = self.si.getControlSpace()
        self.control_sampler = self.control_space.allocControlSampler()

    @classmethod
    def alloc(cls, si):
        return cls(si)

    @classmethod
    def allocator(cls):
        return oc.DirectedControlSamplerAllocator(cls.alloc)

    def sampleTo(self, control_out, previous_control, state, target_out):
        # how do we do this?
        min_step_count = self.si.getMinControlDuration()
        max_step_count = self.si.getMaxControlDuration()
        self.control_sampler.sample(control_out)
        return np.uint32(np.random.random_integers(min_step_count, max_step_count))
