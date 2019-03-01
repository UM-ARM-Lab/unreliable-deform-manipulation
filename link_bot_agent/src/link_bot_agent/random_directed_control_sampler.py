import numpy as np
from ompl import control as oc

from link_bot_agent.my_directed_control_sampler import MyDirectedControlSampler


class RandomDirectedControlSampler(MyDirectedControlSampler):
    states_sampled_at = []

    def __init__(self, si, linear_tf_model):
        super(RandomDirectedControlSampler, self).__init__(si, "random")
        self.linear_tf_model = linear_tf_model

    def sampleTo(self, sampler, control, state, target):
        o = np.ndarray((self.si.getStateDimension(), 1))
        o[0, 0] = state[0]
        o[1, 0] = state[1]
        speed = np.random.uniform(0, 1)
        angle = np.random.uniform(-np.pi, np.pi)
        u = np.array([[np.cos(angle) * speed], [np.sin(angle) * speed]])
        o_next = self.linear_tf_model.simple_predict(o, u)
        control[0] = u[0, 0]
        control[1] = u[1, 0]
        target[0] = o_next[0, 0]
        target[1] = o_next[1, 0]
        duration_steps = 1

        RandomDirectedControlSampler.states_sampled_at.append(state)

        return duration_steps

    @classmethod
    def alloc(cls, si, linear_tf_model):
        return cls(si, linear_tf_model)

    @classmethod
    def allocator(cls, linear_tf_model):
        def partial(si):
            return cls.alloc(si, linear_tf_model)

        return oc.DirectedControlSamplerAllocator(partial)
