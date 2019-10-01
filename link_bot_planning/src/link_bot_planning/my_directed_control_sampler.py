import ompl.util as ou
from ompl import control as oc


class MyDirectedControlSampler(oc.DirectedControlSampler):
    states_sampled_at = []

    def __init__(self, si, action_selector, name):
        super(MyDirectedControlSampler, self).__init__(si)
        self.si = si
        self.name_ = name
        self.action_selector = action_selector
        self.rng_ = ou.RNG()

    @classmethod
    def reset(cls):
        cls.states_sampled_at = []

    @classmethod
    def alloc(cls, si, action_selector):
        return cls(si, action_selector)

    @classmethod
    def allocator(cls, action_selector):
        def partial(si):
            return cls.alloc(si, action_selector)

        return oc.DirectedControlSamplerAllocator(partial)
