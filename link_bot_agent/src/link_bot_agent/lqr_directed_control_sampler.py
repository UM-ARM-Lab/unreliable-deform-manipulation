import numpy as np
from ompl import control as oc
import control
from link_bot_agent.my_directed_control_sampler import MyDirectedControlSampler


class LQRDirectedControlSampler(MyDirectedControlSampler):

    def __init__(self, si, linear_tf_model, max_v):
        super(LQRDirectedControlSampler, self).__init__(si, "LQR")
        self.linear_tf_model = linear_tf_model
        _, self.state_matrix, self.control_matrix, _ = self.linear_tf_model.get_ABCD()
        Q = np.eye(self.linear_tf_model.M)
        R = np.eye(self.linear_tf_model.L)
        self.K, _, _ = control.lqr(self.state_matrix, self.control_matrix, Q, R)
        self.max_v = max_v

    def sampleTo(self, sampler, control, state, target):
        duration_steps = 1
        start_o = np.ndarray((self.si.getStateDimension(), 1))
        start_o[0, 0] = state[0]
        start_o[1, 0] = state[1]
        target_o = np.ndarray((self.si.getStateDimension(), 1))
        target_o[0, 0] = target[0]
        target_o[1, 0] = target[1]
        u = np.dot(-self.K, (start_o - target_o))
        if np.linalg.norm(start_o - target_o) < 1e-3:
            control[0] = 0
            control[1] = 0
            return duration_steps

        if np.linalg.norm(u) < 1e-3:
            raise RuntimeError(
                "Controller is stuck. Cannot progress from {} to {}".format(np.array2string(start_o.squeeze()),
                                                                            np.array2string(target_o.squeeze())))
        u = u / np.linalg.norm(u) * self.max_v
        o_next = self.linear_tf_model.simple_predict(start_o, u)

        LQRDirectedControlSampler.states_sampled_at.append(state)

        control[0] = u[0, 0]
        control[1] = u[1, 0]
        target[0] = o_next[0, 0]
        target[1] = o_next[1, 0]
        return duration_steps

    @classmethod
    def alloc(cls, si, linear_tf_model, max_v):
        return cls(si, linear_tf_model, max_v)

    @classmethod
    def allocator(cls, linear_tf_model, max_v):
        def partial(si):
            return cls.alloc(si, linear_tf_model, max_v)

        return oc.DirectedControlSamplerAllocator(partial)
