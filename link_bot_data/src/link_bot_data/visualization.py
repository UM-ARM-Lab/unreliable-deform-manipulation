import numpy as np
from link_bot_pycommon.link_bot_pycommon import vector_to_points_2d


def plot_rope_configuration(ax, rope_configuration, linewidth=None, linestyle=None, s=1, label=None, scatt=True, **kwargs):
    xs, ys = vector_to_points_2d(rope_configuration)
    if scatt:
        ax.scatter(xs, ys, s=s, **kwargs)
    return ax.plot(xs, ys, linewidth=linewidth, label=label, linestyle=linestyle, **kwargs)


def my_arrow(xs, ys, us, vs):
    xs = np.array(xs)
    ys = np.array(ys)
    us = np.array(us)
    vs = np.array(vs)

    scale = 0.1
    ks = np.sqrt(np.square(us) + np.square(vs)) * scale
    rs = vs / us
    denominators = np.sqrt(np.square(rs) + 1)
    v_primes = ks / denominators
    u_primes = v_primes / rs

    return ([xs, xs + us], [ys, ys + vs]), \
           ([xs + us, xs + us - v_primes * np.sign(us)], [ys + vs, ys + vs - u_primes * np.sign(us)]), \
           ([xs + us, xs + us + u_primes * np.sign(vs)], [ys + vs, ys + vs - v_primes * np.sign(vs)])


def plot_arrow(ax, xs, ys, us, vs, **kwargs):
    xys1, xys2, xys3 = my_arrow(xs, ys, us, vs)
    lines = []
    lines.append(ax.plot(xys1[0], xys1[1], **kwargs)[0])
    lines.append(ax.plot(xys2[0], xys2[1], **kwargs)[0])
    lines.append(ax.plot(xys3[0], xys3[1], **kwargs)[0])
    return lines


def update_arrow(lines, xs, ys, us, vs):
    v_primes, u_primes = my_arrow(xs, ys, us, vs)
    lines[0].set_data([xs, xs + us], [ys, ys + vs])
    lines[1].set_data([xs + us, xs + us - v_primes], [ys + vs, ys + vs - u_primes])
    lines[2].set_data([xs + us, xs + us - u_primes], [ys + vs, ys + vs + v_primes])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure()
    ax = plt.gca()
    plot_arrow(ax, 0, 0, 4, 9)
    plot_arrow(ax, 0, 0, -4, 9)
    plot_arrow(ax, 0, 0, 4, -9)
    plot_arrow(ax, 0, 0, -4, -9)
    plt.axis("equal")
    plt.show()
