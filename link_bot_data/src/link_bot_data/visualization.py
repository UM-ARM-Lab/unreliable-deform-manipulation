def plottable_rope_configuration(rope_configuration):
    config_2d = rope_configuration.reshape(-1, 2)
    xs = config_2d[:, 0]
    ys = config_2d[:, 1]
    return xs, ys


def plot_rope_configuration(ax, rope_configuration, linewidth=None, linestyle=None, s=None, label=None, **kwargs):
    xs, ys = plottable_rope_configuration(rope_configuration)
    ax.scatter(xs, ys, s=s, **kwargs, label=label)
    return ax.plot(xs, ys, linewidth=linewidth, linestyle=linestyle, **kwargs)

# TODO: wrap matplotlib in a way that makes saving/restoring plots very easy
class SavablePlotting:

    def __init__(self):
        self.data = []

    def save_all(self):
        for datum in self.data:
            datum.save()

    def append(self, data):
        self.data.append(data)
