import numpy as np


class Vertex:

    def __init__(self, o):
        self.o = o
        self.f = 1e9
        self.g = 1e9
        self.parent = None

    def __repr__(self):
        ostr = np.array2string(self.o, threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', '')
        return "o:{}, f={} g={}".format(ostr, self.f, self.g)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        o_close = np.linalg.norm(self.o - other.o) < 1e-6
        return o_close

    def __cmp__(self, other):
        return self.f < other.f

    def __lt__(self, other):
        return self.f < other.f


class GzWorldGraph:

    def __init__(self, sdf, resolution=0.1):
        self.sdf = sdf
        self.resolution = resolution
        self.vertices = []

    def neighbors(self, v):
        neighbors = []
        deltas = [
            np.array([[self.resolution], [0]]),
            np.array([[0], [self.resolution]]),
            np.array([[-self.resolution], [0]]),
            np.array([[0], [-self.resolution]]),
            np.array([[self.resolution], [self.resolution]]),
            np.array([[self.resolution], [-self.resolution]]),
            np.array([[-self.resolution], [self.resolution]]),
            np.array([[-self.resolution], [-self.resolution]]),
        ]
        for delta in deltas:
            o_ = v.o + delta
            v_ = None
            vertex_already_exists = False
            for i in self.vertices:
                if np.linalg.norm(i.o - o_) < 1e-6:
                    v_ = i
                    vertex_already_exists = True
                    break
            if not vertex_already_exists:
                v_ = Vertex(o_)
                self.vertices.append(v_)

            if not self.occupied(o_):
                neighbors.append(v_)

        return neighbors

    def occupied(self, o):
        return self.sdf(o)
