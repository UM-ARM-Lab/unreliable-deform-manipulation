from __future__ import print_function
import numpy as np

import heapq


def reconstruct_path(current):
    total_path = [current]
    while True:
        current = current.parent
        if current is None:
            break
        total_path.append(current)
    total_path = total_path[-1::-1]
    return total_path


def near(v1, v2):
    d = np.linalg.norm(np.array(v1.o) - np.array(v2.o))
    return d < 0.1


class AStar:

    def __init__(self, graph, h):
        self.graph = graph
        self.h = h

    def shortest_path(self, start, goal):
        closed_set = []
        open_set = [start]
        start.g = 0
        start.f = self.h(start, goal)

        while len(open_set) != 0:
            # pop the element with the lowest f value
            current = heapq.heappop(open_set)
            # print("EXPANDING", current)
            for o in open_set:
                if o.f < current.f:
                    current = o

            if near(current, goal):
                print("Done!", current, goal)
                return reconstruct_path(current)

            closed_set.append(current)

            for neighbor in self.graph.neighbors(current):
                if neighbor in closed_set:
                    continue
                tentative_g = current.g + self.graph.resolution

                if neighbor.g <= tentative_g:
                    continue

                neighbor.g = tentative_g
                neighbor.f = neighbor.g + self.h(neighbor, goal)
                neighbor.parent = current
                # yes this will lead to duplicates
                heapq.heappush(open_set, neighbor)
                # print("pushing neighbor", neighbor)

        raise ValueError("NO POSSIBLE PATH")
