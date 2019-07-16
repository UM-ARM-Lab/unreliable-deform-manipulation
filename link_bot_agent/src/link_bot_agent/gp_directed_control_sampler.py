from time import time
import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import ompl.base as ob
from PIL import Image
from matplotlib.lines import Line2D
from ompl import control as oc


def inv_sample():
    # v = np.random.uniform(0, 1)
    v = 1
    theta = np.random.uniform(-np.pi, np.pi)
    return np.array([[np.cos(theta) * v, np.sin(theta) * v]])


def plot(state_space, control_space, planner_data, sdf, start, goal, path, controls, arena_size):
    plt.figure()
    max = np.max(np.flipud(sdf.T))
    img = Image.fromarray(np.uint8(np.flipud(sdf.T) / max * 256))
    small_sdf = img.resize((80, 80))
    plt.imshow(small_sdf, extent=[-arena_size, arena_size, -arena_size, arena_size])

    print(len(GPDirectedControlSampler.states_sampled_at))
    print(planner_data.numVertices())

    for state_sampled_at in GPDirectedControlSampler.states_sampled_at:
        xs = [state_sampled_at[0, 0], state_sampled_at[0, 2], state_sampled_at[0, 4]]
        ys = [state_sampled_at[0, 1], state_sampled_at[0, 3], state_sampled_at[0, 5]]
        plt.plot(xs, ys, label='sampled states', linewidth=0.5, c='r', alpha=0.5, zorder=1)

    plt.scatter(start[0, 0], start[0, 1], label='start', s=100, c='c', zorder=1)
    plt.scatter(goal[0, 0], goal[0, 1], label='goal', s=100, c='g', zorder=1)
    for path_i in path:
        xs = [path_i[0], path_i[2], path_i[4]]
        ys = [path_i[1], path_i[3], path_i[5]]
        plt.plot(xs, ys, label='final path', linewidth=4, c='m', alpha=0.75, zorder=4)
    plt.quiver(path[:-1, 4], path[:-1, 5], controls[:, 0], controls[:, 1], width=0.002, zorder=5, color='k')

    for vertex_index in range(planner_data.numVertices()):
        v = planner_data.getVertex(vertex_index)
        # draw the configuration of the rope
        s = v.getState()
        edges_map = ob.mapUintToPlannerDataEdge()

        np_s = state_space.to_numpy(s)
        plt.scatter(np_s[0, 0], np_s[0, 1], s=15, c='orange', zorder=2, alpha=0.5, label='tail')

        if len(edges_map.keys()) == 0:
            xs = [np_s[0, 0], np_s[0, 2], np_s[0, 4]]
            ys = [np_s[0, 1], np_s[0, 3], np_s[0, 5]]
            plt.plot(xs, ys, linewidth=1, c='orange', alpha=0.2, zorder=2, label='full rope')

        planner_data.getEdges(vertex_index, edges_map)
        for vertex_index2 in edges_map.keys():
            v2 = planner_data.getVertex(vertex_index2)
            s2 = v2.getState()
            np_s2 = state_space.to_numpy(s2)
            plt.plot([np_s[0, 0], np_s2[0, 0]], [np_s[0, 1], np_s2[0, 1]], c='white', label='tree', zorder=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-arena_size, arena_size])
    plt.ylim([-arena_size, arena_size])
    plt.axis("equal")

    custom_lines = [
        Line2D([0], [0], color='y', lw=1),
        Line2D([0], [0], color='g', lw=1),
        Line2D([0], [0], color='m', lw=1),
        Line2D([0], [0], color='k', lw=1),
        Line2D([0], [0], color='orange', lw=1),
        Line2D([0], [0], color='white', lw=1),
    ]

    plt.legend(custom_lines, ['start', 'goal', 'final path', 'controls', 'full rope', 'search tree'])


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
        self.state_space = self.si.getStateSpace()
        self.control_space = self.si.getControlSpace()
        self.n_state = self.state_space.getDimension()
        self.n_control = self.control_space.getDimension()
        self.min_steps = int(self.si.getMinControlDuration())
        self.max_steps = int(self.si.getMaxControlDuration())
        self.inv_gp_model.initialize_rng(self.min_steps, self.max_steps)

    @classmethod
    def alloc(cls, si, fwd_gp_model, inv_gp_model, max_v):
        return cls(si, fwd_gp_model, inv_gp_model, max_v)

    @classmethod
    def allocator(cls, fwd_gp_model, inv_gp_model, max_v):
        def partial(si):
            return cls.alloc(si, fwd_gp_model, inv_gp_model, max_v)

        return oc.DirectedControlSamplerAllocator(partial)

    def sampleTo(self, control_out, previous_control, state, target_out):
        np_s = self.state_space.to_numpy(state)
        np_target = self.state_space.to_numpy(target_out)

        self.states_sampled_at.append(np_target)

        # u, duration_steps_float = self.inv_gp_model.inv_act(np_s, np_target, self.max_v)
        u, duration_steps_float = self.inv_gp_model.dumb_inv_act(self.fwd_gp_model, np_s, np_target, self.max_v)

        duration_steps = max(int(duration_steps_float), 1)
        for i in range(duration_steps):
            np_s_next = self.fwd_gp_model.fwd_act(np_s, u)
            # use the already allocated target_out OMPL state for the intermediate states
            self.state_space.from_numpy(np_s_next, target_out)
            if not self.si.isValid(target_out):
                duration_steps = i
                break

            np_s = np_s_next

        # check validity
        if not self.si.isValid(target_out):
            duration_steps = 0

        self.control_space.from_numpy(u, control_out)
        self.state_space.from_numpy(np_s_next, target_out)

        return duration_steps
