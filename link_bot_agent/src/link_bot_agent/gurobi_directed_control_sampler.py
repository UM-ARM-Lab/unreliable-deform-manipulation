import numpy as np
from ompl import control as oc

from link_bot_agent.my_directed_control_sampler import MyDirectedControlSampler


class GurobiDirectedControlSampler(MyDirectedControlSampler):

    def __init__(self, si, gurobi_solver):
        super(GurobiDirectedControlSampler, self).__init__(si, "Gurobi")
        self.gurobi_solver = gurobi_solver

    def sampleTo(self, sampler, control, state, target):
        o = np.ndarray((self.si.getStateDimension(), 1))
        og = np.ndarray((self.si.getStateDimension(), 1))
        o[0, 0] = state[0]
        o[1, 0] = state[1]
        og[0, 0] = target[0]
        og[1, 0] = target[1]
        u, o_next = self.gurobi_solver.act(o, og)
        control[0] = u[0, 0, 0]
        control[1] = u[0, 0, 1]
        target[0] = o_next[0, 0]
        target[1] = o_next[1, 0]
        duration_steps = 1

        GurobiDirectedControlSampler.states_sampled_at.append(state)

        return duration_steps

    @classmethod
    def alloc(cls, si, gurobi_solver):
        return cls(si, gurobi_solver)

    @classmethod
    def allocator(cls, gurobi_solver):
        def partial(si):
            return cls.alloc(si, gurobi_solver)

        return oc.DirectedControlSamplerAllocator(partial)
