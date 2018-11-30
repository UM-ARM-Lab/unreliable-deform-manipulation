from __future__ import print_function


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current_ = came_from[current]
        total_path.append(current_)
        current = current_
    return total_path[-1::-1]


class AStar:

    def __init__(self, graph, h):
        self.graph = graph
        self.h = h

    def shortest_path(self, start, goal):
        closed_set = []
        open_set = [start]
        came_from = {}
        start.g = 0
        start.f = self.h(start, goal)

        def _vertex_set_contains(set, v):
            for v2 in set:
                if v == v2:
                    return True

        while len(open_set) != 0:
            current = open_set[0]
            for o in open_set:
                if o.f < current.f:
                    current = o
            open_set.remove(current)

            if current == goal:
                print(current, goal)
                return reconstruct_path(came_from, current)

            closed_set.append(current)

            print("EXPANDING", current)
            for neighbor in self.graph.vertices[current]:
                if _vertex_set_contains(closed_set, neighbor):
                    continue
                tentative_g = current.g + self.h(current, neighbor)
                if not _vertex_set_contains(open_set, neighbor):
                    open_set.append(neighbor)
                elif neighbor.g <= tentative_g:
                    continue

                neighbor.g = tentative_g
                neighbor.f = neighbor.g + self.h(neighbor, goal)
                came_from[neighbor] = current

        raise ValueError("NO POSSIBLE PATH")
