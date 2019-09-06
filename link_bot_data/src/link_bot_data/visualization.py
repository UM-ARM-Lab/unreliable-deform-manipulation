def plottable_rope_configuration(rope_configuration):
    xs = [rope_configuration[0], rope_configuration[2], rope_configuration[4]]
    ys = [rope_configuration[1], rope_configuration[3], rope_configuration[5]]
    return xs, ys


def plot_rope_configuration(ax, rope_configuration, **kwargs):
    xs, ys = plottable_rope_configuration(rope_configuration)
    return ax.plot(xs, ys, **kwargs)
