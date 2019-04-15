import numpy as np
from sklearn.decomposition import PCA
import ompl.util as ou
from ompl import control as oc

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MyDirectedControlSampler(oc.DirectedControlSampler):
    states_sampled_at = []

    pca_2d = PCA(n_components=2)

    def __init__(self, si, action_selector, name):
        super(MyDirectedControlSampler, self).__init__(si)
        self.si = si
        self.name_ = name
        self.action_selector = action_selector
        self.rng_ = ou.RNG()

    def just_d_multi_act(self, o, og):
        return self.action_selector.just_dmulti_act(o, og)

    def dual_multi_act(self, o_d, o_k, o_d_goal):
        return self.action_selector.dual_multi_act(o_d, o_k, o_d_goal)

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

    @classmethod
    def plot_2d(cls, start, goal, path, pca=True):
        raise NotImplementedError("this function is wrong")
        if pca:
            MP = start.shape[0]
            sampled_points = np.ndarray((len(cls.states_sampled_at), MP))
            for i, (s, p) in enumerate(zip(sampled_points, cls.states_sampled_at)):
                for i in range(MP):
                    s[i] = p[i]
            points_2d = MyDirectedControlSampler.pca_2d.fit_transform(sampled_points)
            start = MyDirectedControlSampler.pca_2d.transform(start.T).T
            goal = MyDirectedControlSampler.pca_2d.transform(goal.T).T
            path = MyDirectedControlSampler.pca_2d.transform(path)
        else:
            points_2d = np.ndarray((len(cls.states_sampled_at), 2))
            for i, (s, p) in enumerate(zip(points_2d, cls.states_sampled_at)):
                for i in range(2):
                    s[i] = p[i]

        plt.figure()
        plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1)
        plt.scatter(start[0, 0], start[1, 0], label='start', s=100, c='blue')
        plt.scatter(goal[0, 0], goal[1, 0], label='goal', s=100, c='green')
        plt.scatter(path[:, 0], path[:, 1], label='path', s=10, c='orange')
        plt.xlabel("o0")
        plt.ylabel("o1")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.legend()

    @classmethod
    def plot_dual_sdf(cls, sdf, start, goal, d_path, k_path, controls):
        o_d_points = np.ndarray((len(cls.states_sampled_at), 2))
        o_k_points = np.ndarray((len(cls.states_sampled_at), 2))
        for i, (o_d, o_k, p) in enumerate(zip(o_d_points, o_k_points, cls.states_sampled_at)):
            for i in range(2):
                o_d[i] = p[0][i]
            for i in range(2):
                o_k[i] = p[1][i]

        plt.figure()
        plt.imshow(np.flipud(sdf.T), extent=[-5.5, 5.5, -5.5, 5.5])
        plt.scatter(o_d_points[:, 0], o_d_points[:, 1], s=1, label='o_d')
        plt.scatter(o_k_points[:, 0], o_k_points[:, 1], s=1, label='o_k')
        plt.scatter(start[0, 0], start[1, 0], label='start', s=100)
        plt.scatter(goal[0, 0], goal[1, 0], label='goal', s=100)
        plt.plot(d_path[:, 0], d_path[:, 1], label='d path', linewidth=3, c='m')
        plt.plot(k_path[:, 0], k_path[:, 1], label='k path', linewidth=3, c='y')
        plt.quiver(d_path[:-1, 0], d_path[:-1, 1], controls[:, 0, 0], controls[:, 0, 1])
        plt.xlabel("o0")
        plt.ylabel("o1")
        plt.axis("equal")
        plt.legend()

    @classmethod
    def plot_controls(cls, controls):
        plt.figure()
        plt.plot(controls[:, 0, 0], controls[:, 0, 1])
        plt.xlabel("u0")
        plt.ylabel("u1")
        plt.title("control inputs")
