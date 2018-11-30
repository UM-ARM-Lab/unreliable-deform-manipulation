#!/usr/bin/env python

import unittest
import numpy as np

from link_bot_agent import a_star
from link_bot_agent import gz_world
from link_bot_notebooks import base_model


def h(v1, v2):
    return np.linalg.norm(np.array(v1.o) - np.array(v2.o))


class TestModel(base_model.BaseModel):

    def __init__(self, dt, N, M, L):
        base_model.BaseModel.__init__(self, N, M, L)
        self.dt = dt

    def predict(self, o, u):
        next_o = o + np.dot(self.dt * np.eye(2), u)
        return next_o


class AStarTest(unittest.TestCase):

    def test_graph(self):
        model = base_model.BaseModel(N=6, M=2, L=2)
        graph = gz_world.GzWorldGraph(model, resolution=0.1, bounds=[-4, 4, -4, 4], num_vertices=100)
        self.assertEqual(len(graph.vertices), 100)

    def test_a_star(self):
        dt = 0.1
        model = TestModel(dt, N=6, M=2, L=2)
        graph = gz_world.GzWorldGraph(model, resolution=0.1, bounds=[-5, 5, -5, 5], dt=dt, num_vertices=1000)
        planner = a_star.AStar(graph, h)
        o = gz_world.Vertex(np.array([[0], [0]]), 0, 0)
        og = gz_world.Vertex(np.array([[0], [2]]), 0, 2)
        shortest_path = planner.shortest_path(o, og)
        print(shortest_path)
        # self.assertEqual(shortest_path, [gz_world.Vertex(np.array([[0], [0]]), gz_world.Vertex(np.array([[0], [1]]), gz_world.Vertex(0, 2)])


if __name__ == '__main__':
    unittest.main()
