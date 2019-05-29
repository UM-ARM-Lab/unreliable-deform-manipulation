import matplotlib.pyplot as plt
import numpy as np
from ompl import control as oc
import ompl.util as ou


class GPDirectedControlSampler(oc.DirectedControlSampler):
    states_sampled_at = []

    def __init__(self, si, fwd_gp_model, inv_gp_model):
        super(GPDirectedControlSampler, self).__init__(si)
        self.si = si
        self.name_ = 'gp_dcs'
        self.rng_ = ou.RNG()
        self.fwd_gp_model = fwd_gp_model
        self.inv_gp_model = inv_gp_model

    @classmethod
    def alloc(cls, si, fwd_gp_model, inv_gp_model):
        return cls(si, fwd_gp_model, inv_gp_model)

    @classmethod
    def allocator(cls, fwd_gp_model, inv_gp_model):
        def partial(si):
            return cls.alloc(si, fwd_gp_model, inv_gp_model)

        return oc.DirectedControlSamplerAllocator(partial)

    @classmethod
    def reset(cls):
        cls.states_sampled_at = []

    def sampleTo(self, control_out, previous_control, state, target_out):
        # we return 0 to indicate no duration when LQR gives us a control that takes us into collision
        # this will cause the RRT to throw out this motion
        n_state = self.si.getStateSpace().getDimension()
        n_control = self.si.getControlSpace().getDimension()
        s = np.ndarray((n_state, 1))
        s_target = np.ndarray((n_state, 1))
        for i in range(n_state):
            s[i, 0] = state[i]
            s_target[i, 0] = target_out[i]

        u = self.inv_gp_model.inv_act(s, s_target)
        s_next = self.fwd_gp_model.fwd_act(s, u)

        for i in range(n_control):
            control_out[i] = u[0, i]
        for i in range(n_state):
            target_out[i] = s_next[0, i]

        GPDirectedControlSampler.states_sampled_at.append(state)

        # check validity
        if not self.si.isValid(target_out):
            return 0

        duration_steps = 1
        return duration_steps

    @classmethod
    def plot(cls, sdf, start, goal, path, controls, arena_size):
        head = np.ndarray((len(cls.states_sampled_at), 2))
        for i in range(len(cls.states_sampled_at)):
            head[i, 0] = cls.states_sampled_at[i][0]
            head[i, 1] = cls.states_sampled_at[i][1]

        plt.figure()
        plt.imshow(np.flipud(sdf.T), extent=[-arena_size, arena_size, -arena_size, arena_size])
        plt.scatter(head[:, 0], head[:, 1], s=1, label='head')
        plt.scatter(start[0, 0], start[1, 0], label='start', s=100)
        plt.scatter(goal[0, 0], goal[1, 0], label='goal', s=100)
        plt.plot(path[:, 0], path[:, 1], label='d path', linewidth=3, c='m')
        plt.quiver(path[:-1, 0], path[:-1, 1], controls[:, 0, 0], controls[:, 0, 1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.legend()
