import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from ompl import base as ob

from link_bot_planning.gp_directed_control_sampler import GPDirectedControlSampler
from link_bot_planning.state_spaces import to_numpy


def plot_rrt(rrt, show=True):
    si = rrt.getSpaceInformation()
    planner_data = ob.PlannerData(si)
    rrt.getPlannerData(planner_data)

    graphml = planner_data.printGraphML()
    graphml = graphml.replace("string", "double")
    f = open("graph.graphml", 'w')
    f.write(graphml)
    f.close()

    # Load the graphml data using graph-tool
    graph = gt.load_graph("graph.graphml", fmt="xml")
    edgeweights = graph.edge_properties["weight"]

    # Write some interesting statistics
    avgdeg, stddevdeg = gt.vertex_average(graph, "total")
    avgwt, stddevwt = gt.edge_average(graph, edgeweights)

    print("---- PLANNER DATA STATISTICS ----")
    print(str(graph.num_vertices()) + " vertices and " + str(graph.num_edges()) + " edges")
    print("Average vertex degree (in+out) = " + str(avgdeg) + "  St. Dev = " + str(stddevdeg))
    print("Average edge weight = " + str(avgwt) + "  St. Dev = " + str(stddevwt))

    _, hist = gt.label_components(graph)
    print("Strongly connected components: " + str(len(hist)))

    # Make the graph undirected (for weak components, and a simpler drawing)
    graph.set_directed(False)
    _, hist = gt.label_components(graph)
    print("Weakly connected components: " + str(len(hist)))

    # Plotting the graph
    gt.remove_parallel_edges(graph)  # Removing any superfluous edges

    edgeweights = graph.edge_properties["weight"]
    colorprops = graph.new_vertex_property("string")
    vertexsize = graph.new_vertex_property("double")

    start = -1
    goal = -1

    for v in range(graph.num_vertices()):

        # Color and size vertices by type: start, goal, other
        if planner_data.isStartVertex(v):
            start = v
            colorprops[graph.vertex(v)] = "cyan"
            vertexsize[graph.vertex(v)] = 10
        elif planner_data.isGoalVertex(v):
            goal = v
            colorprops[graph.vertex(v)] = "green"
            vertexsize[graph.vertex(v)] = 10
        else:
            colorprops[graph.vertex(v)] = "yellow"
            vertexsize[graph.vertex(v)] = 5

    # default edge color is black with size 0.5:
    edgecolor = graph.new_edge_property("string")
    edgesize = graph.new_edge_property("double")
    for e in graph.edges():
        edgecolor[e] = "black"
        edgesize[e] = 0.5

    # using A* to find shortest path in planner data
    if start != -1 and goal != -1:
        _, pred = gt.astar_search(graph, graph.vertex(start), edgeweights)

        # Color edges along shortest path red with size 3.0
        v = graph.vertex(goal)
        while v != graph.vertex(start):
            p = graph.vertex(pred[v])
            for e in p.out_edges():
                if e.target() == v:
                    edgecolor[e] = "red"
                    edgesize[e] = 2.0
            v = p

    # Writing graph to file:
    # pos indicates the desired vertex positions, and pin=True says that we
    # really REALLY want the vertices at those positions
    pos = graph.new_vertex_property("double")
    # for coord in graph.vertex_properties['coords']:
    #     pass
    # pos.append([float(d) for d in coord.split(",")])
    gt.graph_draw(graph, pos, pin=True, vertex_size=vertexsize, vertex_fill_color=colorprops,
                  edge_pen_width=edgesize, edge_color=edgecolor)

    if show:
        img = plt.imread("graph.png")
        plt.imshow(img)
        plt.show()
        input("waiting...")


def flatten_plan(np_controls, np_duration_steps_int):
    # flatten and combine np_controls and durations
    np_controls_flat = []
    for control, duration in zip(np_controls, np_duration_steps_int):
        for i in range(duration):
            np_controls_flat.append(control)
    np_controls_flat = np.array(np_controls_flat)
    return np_controls_flat


def flatten_plan_dt(np_controls, np_durations, dt):
    # flatten and combine np_controls and durations
    np_duration_steps_int = (np_durations / dt).astype(np.int)
    np_controls_flat = []
    for control, duration in zip(np_controls, np_duration_steps_int):
        for i in range(duration):
            np_controls_flat.append(control)
    np_controls_flat = np.array(np_controls_flat)
    return np_controls_flat


def plot(planner_data, sdf, start, goal, path, controls, n_state, extent):
    plt.figure()
    plt.imshow(np.flipud(sdf) > 0, extent=extent)

    for state_sampled_at in GPDirectedControlSampler.states_sampled_at:
        xs = [state_sampled_at[0, 0], state_sampled_at[0, 2], state_sampled_at[0, 4]]
        ys = [state_sampled_at[0, 1], state_sampled_at[0, 3], state_sampled_at[0, 5]]
        plt.plot(xs, ys, label='sampled states', linewidth=0.5, c='b', alpha=0.5, zorder=1)

    plt.scatter(start[0, 0], start[0, 1], label='start', s=100, c='r', zorder=1)
    plt.scatter(goal[0], goal[1], label='goal', s=100, c='g', zorder=1)
    for path_i in path:
        xs = [path_i[0], path_i[2], path_i[4]]
        ys = [path_i[1], path_i[3], path_i[5]]
        plt.plot(xs, ys, label='final path', linewidth=2, c='cyan', alpha=0.75, zorder=4)
    plt.quiver(path[:-1, 4], path[:-1, 5], controls[:, 0], controls[:, 1], width=0.004, zorder=5, color='k', alpha=0.5)

    for vertex_index in range(planner_data.numVertices()):
        v = planner_data.getVertex(vertex_index)
        # draw the configuration of the rope
        s = v.getState()
        edges_map = ob.mapUintToPlannerDataEdge()

        np_s = to_numpy(s, n_state)
        plt.scatter(np_s[0, 0], np_s[0, 1], s=15, c='orange', zorder=2, alpha=0.5, label='tail')

        if len(edges_map.keys()) == 0:
            xs = [np_s[0, 0], np_s[0, 2], np_s[0, 4]]
            ys = [np_s[0, 1], np_s[0, 3], np_s[0, 5]]
            plt.plot(xs, ys, linewidth=1, c='orange', alpha=0.2, zorder=2, label='full rope')

        planner_data.getEdges(vertex_index, edges_map)
        for vertex_index2 in edges_map.keys():
            v2 = planner_data.getVertex(vertex_index2)
            s2 = v2.getState()
            np_s2 = to_numpy(s2, n_state)
            plt.plot([np_s[0, 0], np_s2[0, 0]], [np_s[0, 1], np_s2[0, 1]], c='white', label='tree', zorder=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.xlim(extent[0:2])
    plt.ylim(extent[2:4])

    custom_lines = [
        Line2D([0], [0], color='b', lw=1),
        Line2D([0], [0], color='r', lw=1),
        Line2D([0], [0], color='g', lw=1),
        Line2D([0], [0], color='cyan', lw=1),
        Line2D([0], [0], color='k', lw=1),
        Line2D([0], [0], color='orange', lw=1),
        Line2D([0], [0], color='orange', lw=1),
        Line2D([0], [0], color='white', lw=1),
    ]

    plt.legend(custom_lines,
               ['sampled rope configurations', 'start', 'goal', 'final path', 'controls', 'full rope', 'search tree'])
    plt.show()
