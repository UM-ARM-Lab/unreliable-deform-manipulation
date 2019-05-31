import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
from matplotlib.lines import Line2D
from ompl import control as oc

from link_bot_gaussian_process.link_bot_gp import LinkBotGP

class GPDirectedControlSampler(oc.DirectedControlSampler):
    states_sampled_at = []

    def __init__(self, si, fwd_gp_model, inv_gp_model, max_v):
        super(GPDirectedControlSampler, self).__init__(si)
        self.si = si
        self.name_ = 'gp_dcs'
        self.rng_ = ou.RNG()
        self.max_v = max_v
        self.fwd_gp_model = fwd_gp_model
        self.inv_gp_model = inv_gp_model

    @classmethod
    def alloc(cls, si, fwd_gp_model, inv_gp_model, max_v):
        return cls(si, fwd_gp_model, inv_gp_model, max_v)

    @classmethod
    def allocator(cls, fwd_gp_model, inv_gp_model, max_v):
        def partial(si):
            return cls.alloc(si, fwd_gp_model, inv_gp_model, max_v)

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

        u = self.inv_gp_model.inv_act(s, s_target, self.max_v)
        # u = self.inv_sample()
        # this u will be the [cos, sin, mag] representation so we have to convert first
        u = LinkBotGP.convert_u(u)
        s_next = self.fwd_gp_model.fwd_act(s, u)

        for i in range(n_control):
            control_out[i] = u[0, i]
        for i in range(n_state):
            target_out[i] = s_next[0, i]

        GPDirectedControlSampler.states_sampled_at.append(s)

        # check validity
        if not self.si.isValid(target_out):
            return 0

        duration_steps = 1
        return duration_steps

    @classmethod
    def plot(cls, sdf, start, goal, path, controls, arena_size):
        plt.figure()

        plt.imshow(np.flipud(sdf.T), extent=[-arena_size, arena_size, -arena_size, arena_size])

        for s in cls.states_sampled_at:
            # draw the configuration of the rope
            # plt.plot([s[0, 0], s[2, 0], s[4, 0]], [s[1, 0], s[3, 0], s[5, 0]], linewidth=1, c='k')
            plt.scatter(s[0, 0], s[1, 0], s=1, c='r')
            plt.scatter(s[4, 0], s[5, 0], s=1, c='orange')

        plt.scatter(start[0, 0], start[1, 0], label='start', s=100, c='y')
        plt.scatter(goal[0, 0], goal[1, 0], label='goal', s=100, c='g')
        plt.plot(path[:, 0], path[:, 1], label='tail path', linewidth=3, c='m')
        plt.plot(path[:, 4], path[:, 5], label='head path', linewidth=3, c='k')
        plt.quiver(path[:-1, 0], path[:-1, 1], controls[:, 0], controls[:, 1], width=0.001)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-arena_size, arena_size])
        plt.ylim([-arena_size, arena_size])
        plt.axis("equal")

        custom_lines = [
            Line2D([0], [0], color='r', lw=1),
            Line2D([0], [0], color='orange', lw=1),
            Line2D([0], [0], color='y', lw=1),
            Line2D([0], [0], color='g', lw=1),
            Line2D([0], [0], color='m', lw=1),
            Line2D([0], [0], color='k', lw=1),
        ]

        plt.legend(custom_lines, ['tail', 'head', 'start', 'goal', 'tail path', 'head path'])

    def inv_sample(self):
        # v = np.random.uniform(0, 1)
        v = 1
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([[np.cos(theta) * v, np.sin(theta) * v]])
