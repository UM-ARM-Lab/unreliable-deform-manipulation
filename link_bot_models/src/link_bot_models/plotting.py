from enum import auto

import numpy as np
from link_bot_models import constraint_sdf
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from link_bot_pycommon import link_bot_pycommon


class PlotType(link_bot_pycommon.ArgsEnum):
    none = auto()
    random_individual = auto()
    random_combined = auto()
    true_positives = auto()
    true_negatives = auto()
    false_positives = auto()
    false_negatives = auto()
    interpolate = auto()
    contours = auto()
    animate_contours = auto()


class SavableFigureCollection:

    def __init__(self, figures):
        self.figures = figures

    def save(self, filename):
        for i, f in enumerate(self.figures):
            f.save(filename + '_' + i)


class SavableAnimationWrapper:

    def __init__(self, anim, writer):
        self.anim = anim
        self.writer = writer

    def save(self, filename):
        self.anim.save(filename + '.mp4', writer=self.writer)


class SavableFigure:

    def __init__(self, figure):
        self.figure = figure

    def save(self, filename):
        self.figure.savefig(filename)


def plot_examples(sdf_image, extent, results, subsample=10, title=''):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(sdf_image, extent=extent)

    predicted_xs = [result.predicted_point[0] for result in results[::subsample]]
    predicted_ys = [result.predicted_point[1] for result in results[::subsample]]
    head_xs = [result.rope_configuration[4] for result in results[::subsample]]
    head_ys = [result.rope_configuration[5] for result in results[::subsample]]

    plt.scatter(predicted_xs, predicted_ys, s=5, c='r', label='predicted', zorder=3)
    plt.scatter(head_xs, head_ys, s=5, c='b', label='true', zorder=2)

    for result in results[::subsample]:
        predicted_point = result.predicted_point
        rope_configuration = result.rope_configuration
        plt.plot([predicted_point[0], rope_configuration[4]], [predicted_point[1], rope_configuration[5]], c='k',
                 linewidth=1, zorder=1, alpha=0.1)
    plt.legend()
    return SavableFigure(fig)


def plot_single_example(sdf_data, result):
    fig = plt.figure()
    plt.imshow(sdf_data.image, extent=sdf_data.extent)
    plt.plot(result.rope_configuration[[0, 2, 4]], result.rope_configuration[[1, 3, 5]], label='rope')

    if result.predicted_violated:
        pred_color = 'r'
    else:
        pred_color = 'g'

    if result.true_violated:
        true_color = 'r'
    else:
        true_color = 'g'

    plt.scatter(result.predicted_point[0], result.predicted_point[1], s=5, c=pred_color, label='pred')
    plt.scatter(result.rope_configuration[4], result.rope_configuration[5], s=5, c=true_color, label='true')
    plt.legend()
    return SavableFigure(fig)


def plot_interpolate(sdf_data, sdf_image, model, threshold, title=''):
    fig = plt.figure()
    plt.title(title)
    plt.imshow(sdf_image, extent=sdf_data.extent)

    head_xs = np.linspace(sdf_data.extent[0] + 1, sdf_data.extent[1] - 1, 25)
    head_ys = np.linspace(sdf_data.extent[2] + 1, sdf_data.extent[3] - 1, 25)
    theta_1s = np.linspace(-np.pi, np.pi, 2)
    theta_2s = np.linspace(-np.pi, np.pi, 2)
    grid = np.meshgrid(head_xs, head_ys, theta_1s, theta_2s)
    grid = [g.reshape(-1) for g in grid]
    rope_params = np.vstack(grid).T
    rope_configuration_0 = link_bot_pycommon.make_rope_configuration(*rope_params[0])

    result_0 = constraint_sdf.test_single_prediction(sdf_data, model, threshold, rope_configuration_0)
    red_arr = np.array([[0.9, 0.2, 0.2]])
    green_arr = np.array([[0.2, 0.9, 0.2]])
    small_arr = np.array([25])
    big_arr = np.array([100])
    color_0 = green_arr if result_0.true_violated == result_0.predicted_violated else red_arr
    head_scatter = plt.scatter(result_0.rope_configuration[4], result_0.rope_configuration[5], s=50, c=color_0, zorder=2)
    prediction_scatter = plt.scatter(result_0.predicted_point[0], result_0.predicted_point[1], s=50, c=color_0, zorder=2)

    xs_0 = [rope_configuration_0[0], rope_configuration_0[2], rope_configuration_0[4]]
    ys_0 = [rope_configuration_0[1], rope_configuration_0[3], rope_configuration_0[5]]
    line = plt.plot(xs_0, ys_0, color='black', linewidth=2, zorder=1)[0]

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    custom_lines = [
        Line2D([0], [0], color='r', lw=1),
        Line2D([0], [0], color='b', lw=1),
    ]
    plt.legend(custom_lines, ['head', 'prediction'])

    def update(t):
        rope_configuration = link_bot_pycommon.make_rope_configuration(*rope_params[t])
        result = constraint_sdf.test_single_prediction(sdf_data, model, threshold, rope_configuration)
        color = green_arr if result.true_violated == result.predicted_violated else red_arr
        sizes = small_arr if result.true_violated == result.predicted_violated else big_arr

        head_scatter.set_offsets(rope_configuration[4:6])
        head_scatter.set_color(color)
        head_scatter.set_sizes(sizes)

        prediction_scatter.set_offsets(result.predicted_point)
        prediction_scatter.set_color(color)
        prediction_scatter.set_sizes(sizes)

        xs = [rope_configuration[0], rope_configuration[2], rope_configuration[4]]
        ys = [rope_configuration[1], rope_configuration[3], rope_configuration[5]]
        line.set_xdata(xs)
        line.set_ydata(ys)

    fps = 100
    T = rope_params.shape[0]
    interval_ms = 10
    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800)
    return SavableAnimationWrapper(anim, writer)


def animate_contours(sdf_data, model, threshold):
    fig = plt.figure()
    plt.title('level sets')
    binary_image = np.asarray(sdf_data.image) > threshold
    plt.imshow(binary_image, extent=sdf_data.extent)

    xmin, xmax, ymin, ymax = sdf_data.extent
    contour_spacing_y = 0.5
    contour_spacing_x = 0.5
    y_range = np.arange(ymin + 0.1, ymax + sdf_data.resolution[0] - 0.1, sdf_data.resolution[0] * contour_spacing_y)
    y = y_range[0]
    head_xs_flat = np.arange(xmin + 0.1, xmax + sdf_data.resolution[1] - 0.1, sdf_data.resolution[1] * contour_spacing_x)
    head_ys_flat = np.ones_like(head_xs_flat) * y

    zeros = np.zeros_like(head_xs_flat)
    rope_configurations = link_bot_pycommon.make_rope_configurations(head_xs_flat, head_ys_flat, zeros, zeros)

    _, predicted_points = model.violated(rope_configurations, sdf_data)

    predicted_xs = predicted_points[:, 0]
    predicted_ys = predicted_points[:, 1]

    head_line = plt.plot(head_xs_flat, head_ys_flat, color='r', linewidth=3, zorder=1)[0]
    predicted_line = plt.plot(predicted_xs, predicted_ys, color='b', linewidth=1, zorder=2)[0]

    custom_lines = [
        Line2D([0], [0], color='r', lw=1),
        Line2D([0], [0], color='b', lw=1),
    ]
    plt.legend(custom_lines, ['head', 'prediction'])

    def update(t):
        y = y_range[t]
        head_ys_flat = np.ones_like(head_xs_flat) * y

        rope_configurations = link_bot_pycommon.make_rope_configurations(head_xs_flat, head_ys_flat, zeros, zeros)

        _, predicted_points = model.violated(rope_configurations, sdf_data)

        predicted_xs = predicted_points[:, 0]
        predicted_ys = predicted_points[:, 1]

        head_line.set_xdata(head_xs_flat)
        head_line.set_ydata(head_ys_flat)
        predicted_line.set_xdata(predicted_xs)
        predicted_line.set_ydata(predicted_ys)

    fps = 100
    duration_s = 10
    T = y_range.shape[0]
    interval_ms = int(duration_s * 1000 / T)
    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800)
    return SavableAnimationWrapper(anim, writer)


def plot_contours(sdf_data, model, threshold):
    fig = plt.figure()
    plt.title('level sets')
    binary_image = np.asarray(sdf_data.image) > threshold
    plt.imshow(binary_image, extent=sdf_data.extent)

    x_min, x_max, y_min, y_max = sdf_data.extent
    contour_spacing_y = 1000
    contour_spacing_x = 1
    y_range = np.arange(y_min, y_max + sdf_data.resolution[0], sdf_data.resolution[0] * contour_spacing_y)
    for y in y_range:
        head_xs_flat = np.arange(x_min, x_max + sdf_data.resolution[1], sdf_data.resolution[1] * contour_spacing_x)
        head_ys_flat = np.ones_like(head_xs_flat) * y

        zeros = np.zeros_like(head_xs_flat)
        rope_configurations = link_bot_pycommon.make_rope_configurations(head_xs_flat, head_ys_flat, zeros, zeros)

        _, predicted_points = model.violated(rope_configurations, sdf_data)

        predicted_xs = predicted_points[:, 0]
        predicted_ys = predicted_points[:, 1]

        red = blue = (y - y_range[0]) / (y_range[-1] - y_range[0])
        plt.plot(head_xs_flat, head_ys_flat, color=(red, 0, 0), linewidth=3, zorder=1)
        plt.plot(predicted_xs, predicted_ys, color=(0, 0, blue), linewidth=1, zorder=2)

    custom_lines = [
        Line2D([0], [0], color='r', lw=1),
        Line2D([0], [0], color='b', lw=1),
    ]
    plt.legend(custom_lines, ['head', 'prediction'])
    return SavableFigure(fig)


def plot_examples_on_fig(fig, x, constraint_labels, threshold, model, draw_correspondences=False):
    sdf_data = x[0]
    rope_configurations = x[1]
    binary_image = sdf_data.image > threshold
    plt.imshow(binary_image, extent=sdf_data.extent)

    ax = fig.gca()

    violated, predicted_points = model.violated(rope_configurations, sdf_data)
    for rope_configuration, predicted_point, constraint_label in zip(rope_configurations, predicted_points, constraint_labels):
        color = 'r' if constraint_label else 'g'
        ax.scatter(rope_configuration[4], rope_configuration[5], c=color, s=10, zorder=3)
        ax.plot(rope_configuration[[0, 2, 4]], rope_configuration[[1, 3, 5]], zorder=2)
        if draw_correspondences:
            ax.plot([rope_configuration[4], predicted_point[0]], [rope_configuration[5], predicted_point[1]], zorder=1, c='k')

    pred_color = ['r' if v else 'g' for v in violated]
    ax.scatter(predicted_points[:, 0], predicted_points[:, 1], c=pred_color, s=10, zorder=4, marker='x')

    return SavableFigure(fig)
