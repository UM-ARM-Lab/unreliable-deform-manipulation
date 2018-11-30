from __future__ import print_function
import numpy as np


class Vertex:

    def __init__(self, o, head_x, head_y):
        self.o = o
        self.head_x = head_x
        self.head_y = head_y
        self.f = 1e9
        self.g = 1e9

    def __repr__(self):
        ostr = np.array2string(self.o, threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', '')
        return "o:{}, h:{},{}, f={} g={}".format(ostr, self.head_x, self.head_y, self.f, self.g)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        print(self, other)
        o_close = np.linalg.norm(self.o - other.o) < 1e-6
        head_close = abs(self.head_x - other.head_x) < 1e-6 and abs(self.head_y - other.head_y) < 1e-6
        return o_close and head_close


class GzWorldGraph:

    def __init__(self, model, resolution=0.1, bounds=None, dt=0.1, num_vertices=1000):

        self.vertices = []
        self.edges = []
        self.dt = dt
        self.resolution = resolution
        self.model = model
        self.NUM_VERTICES = num_vertices

        if bounds is None:
            self.bounds = [-20, 20, -20, 20]
        else:
            self.bounds = bounds

        # initalize occupancy map to all zeros (not occupied)
        height = int((bounds[3] - bounds[2]) / resolution)
        width = int((bounds[1] - bounds[0]) / resolution)
        self.occupancy_grid = np.zeros((height, width), dtype=np.bool)

        # pick an arbitrary starting configuration and procedurally expand all reachable vertices
        o = np.array([[0], [0]])
        head_x = 0
        head_y = 2
        v = Vertex(o, head_x, head_y)
        self.expand(v)

    def in_bounds(self, x, y):
        return self.bounds[0] < x < self.bounds[1] and self.bounds[2] < y < self.bounds[3]

    def compute_occupancy_grid(self, boxes, size=1):
        s = size/2
        e = 1e-3  # small epsilon to make the arange work
        for b in boxes:
            for bx in np.arange(b[0] - s, b[0] + s + e, self.resolution):
                for by in np.arange(b[1] - s, b[1] + s + e, self.resolution):
                    if self.in_bounds(bx, by):
                        intbx = int(b[0])
                        intby = int(b[1])
                        self.occupancy_grid[intby][intbx] = True

    def expand(self, init_v):
        queue = [init_v]
        while len(queue) != 0:
            v = queue.pop(0)
            self.vertices.append(v)
            # stopping condition
            if len(self.vertices) >= self.NUM_VERTICES:
                break

            # sample a bunch of random actions
            potential_actions = 0.5 * np.random.randn(250, 2, 1)
            for u in potential_actions:
                # apply a hard-coded motion model to the head
                new_head_x = v.head_x + self.dt * u[0, 0]
                new_head_y = v.head_y + self.dt * u[1, 0]
                if self.in_bounds(new_head_x, new_head_y):
                    if not self.occupied(new_head_x, new_head_y):
                        new_o = self.model.predict(v.o, u)
                        new_v = Vertex(new_o, new_head_x, new_head_y)
                        queue.append(new_v)

    def occupied(self, x, y):
        intx = int(x)
        inty = int(y)
        occupied = self.occupancy_grid[intx][inty]
        return occupied
