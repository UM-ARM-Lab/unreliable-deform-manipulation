def plottable_rope_configuration(rope_configuration):
    config_2d = rope_configuration.reshape(-1, 2)
    xs = config_2d[:, 0]
    ys = config_2d[:, 1]
    return xs, ys


def plot_rope_configuration(ax, rope_configuration, **kwargs):
    xs, ys = plottable_rope_configuration(rope_configuration)
    return ax.plot(xs, ys, **kwargs)
