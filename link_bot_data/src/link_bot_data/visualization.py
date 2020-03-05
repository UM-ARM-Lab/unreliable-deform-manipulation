from link_bot_pycommon.link_bot_pycommon import vector_to_points_2d


def plot_rope_configuration(ax, rope_configuration, linewidth=None, linestyle=None, s=1, label=None, scatt=True, **kwargs):
    xs, ys = vector_to_points_2d(rope_configuration)
    if scatt:
        ax.scatter(xs, ys, s=s, **kwargs, label=label)
    return ax.plot(xs, ys, linewidth=linewidth, linestyle=linestyle, **kwargs)
