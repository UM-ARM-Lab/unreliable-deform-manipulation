import numpy as np
import ompl.control as oc


class RandomDirectedControlSampler(oc.DirectedControlSampler):

    def __init__(self, si, seed):
        super().__init__(si)
        self.si = si
        self.control_space = self.si.getControlSpace()
        self.control_sampler = self.control_space.allocControlSampler()
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @classmethod
    def allocator(cls, seed):
        def alloc(si):
            return cls(si, seed)

        return oc.DirectedControlSamplerAllocator(alloc)

    def sampleTo(self, control_out, previous_control, state, target_out):
        # how do we do this?
        min_step_count = self.si.getMinControlDuration()
        max_step_count = self.si.getMaxControlDuration()
        self.control_sampler.sample(control_out)
        print(state[0][0], state[0][1], state[0][2])
        return np.uint32(self.rng.random_integers(min_step_count, max_step_count))
