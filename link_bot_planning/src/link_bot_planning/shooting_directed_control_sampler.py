import ompl.util as ou
from ompl import control as oc
import numpy as np

from link_bot_planning.state_spaces import to_numpy, from_numpy


class ShootingDirectedControlSampler(oc.DirectedControlSampler):
    states_sampled_at = []

    def __init__(self, si, fwd_model, max_v, n_samples):
        super(ShootingDirectedControlSampler, self).__init__(si)
        self.si = si
        self.name_ = 'shooting_dcs'
        self.rng_ = ou.RNG()
        self.max_v = max_v
        self.n_samples = n_samples
        self.fwd_model = fwd_model
        self.state_space = self.si.getStateSpace()
        self.control_space = self.si.getControlSpace()
        self.n_state = self.state_space.getDimension()
        self.n_control = self.control_space.getDimension()
        self.min_steps = int(self.si.getMinControlDuration())
        self.max_steps = int(self.si.getMaxControlDuration())

    @classmethod
    def alloc(cls, si, fwd_model, max_v, n_samples):
        return cls(si, fwd_model, max_v, n_samples)

    @classmethod
    def allocator(cls, fwd_model, max_v, n_samples=10):
        def partial(si):
            return cls.alloc(si, fwd_model, max_v, n_samples)

        return oc.DirectedControlSamplerAllocator(partial)

    def sampleTo(self, control_out, previous_control, state, target_out):
        np_s = to_numpy(state, self.n_state)
        np_target = to_numpy(target_out, self.n_state)

        self.states_sampled_at.append(np_target)

        min_distance = np.inf
        min_u = None
        for i in range(self.n_samples):
            theta = np.random.uniform(-np.pi, np.pi)
            u = np.array([[self.max_v * np.cos(theta), self.max_v * np.sin(theta)]])
            print(np_s.shape, u.shape)
            s_next = self.fwd_model.predict(np_s, u)
            distance = np.linalg.norm(s_next - np_target)
            if distance < min_distance:
                min_distance = distance
                min_u = u

        print(np_s.shape, min_u.shape)
        np_s_next = self.fwd_model.predict(np_s, min_u)

        from_numpy(min_u, control_out, self.n_control)
        from_numpy(np_s_next, target_out, self.n_state)

        # check validity
        duration_steps = 1
        if not self.si.isValid(target_out):
            duration_steps = 0

        return duration_steps
