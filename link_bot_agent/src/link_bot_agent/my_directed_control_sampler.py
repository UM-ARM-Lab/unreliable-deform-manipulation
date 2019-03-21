import numpy as np
from sklearn.decomposition import PCA
import ompl.util as ou
from ompl import control as oc

import matplotlib.pyplot as plt


class MyDirectedControlSampler(oc.DirectedControlSampler):
    states_sampled_at = []

    pca_2d = PCA(n_components=2)

    def __init__(self, si, action_selector, name):
        super(MyDirectedControlSampler, self).__init__(si)
        self.si = si
        self.name_ = name
        self.action_selector = action_selector
        self.rng_ = ou.RNG()

    def multi_act(self, o, og):
        return self.action_selector.multi_act(o, og)

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
    def plot_2d(cls, start, goal, path):
        M = start.shape[0]
        sampled_points = np.ndarray((len(cls.states_sampled_at), M))
        for i, (s, p) in enumerate(zip(sampled_points, cls.states_sampled_at)):
            for i in range(M):
                s[i] = p[i]
        points_2d = MyDirectedControlSampler.pca_2d.fit_transform(sampled_points)
        start = MyDirectedControlSampler.pca_2d.transform(start.T).T
        goal = MyDirectedControlSampler.pca_2d.transform(goal.T).T
        path = MyDirectedControlSampler.pca_2d.transform(path)

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
    def plot_controls(cls, controls):
        plt.figure()
        plt.plot(controls[:, 0, 0], controls[:, 0, 1])
        plt.xlabel("u0")
        plt.ylabel("u1")
        plt.title("control inputs")
