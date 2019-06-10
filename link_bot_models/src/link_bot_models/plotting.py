from enum import auto

import numpy as np
from link_bot_models import constraint_model
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from link_bot_pycommon import link_bot_pycommon


class PlotType(link_bot_pycommon.ArgsEnum):
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


def plot_single_example(sdf_image, extent, result):
    fig = plt.figure()
    plt.imshow(sdf_image, extent=extent)
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

    head_xs = np.linspace(-4.95, 4.95, 25)
    head_ys = np.linspace(-4.95, 4.95, 25)
    theta_1s = np.linspace(-np.pi, np.pi, 2)
    theta_2s = np.linspace(-np.pi, np.pi, 2)
    grid = np.meshgrid(head_xs, head_ys, theta_1s, theta_2s)
    grid = [g.reshape(-1) for g in grid]
    rope_params = np.vstack(grid).T
    rope_configuration_0 = link_bot_pycommon.make_rope_configuration(*rope_params[0])

    result_0 = constraint_model.evaluate_single(sdf_data, model, threshold, rope_configuration_0)
    head_scatter = plt.scatter(result_0.rope_configuration[4], result_0.rope_configuration[5], s=50, c='b', zorder=2)
    prediction_scatter = plt.scatter(result_0.predicted_point[0], result_0.predicted_point[1], s=10, c='r', zorder=2)

    xs_0 = [rope_configuration_0[0], rope_configuration_0[2], rope_configuration_0[4]]
    ys_0 = [rope_configuration_0[1], rope_configuration_0[3], rope_configuration_0[5]]
    line = plt.plot(xs_0, ys_0, color='black', linewidth=2, zorder=1)[0]

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-5.1, 5.1])
    plt.ylim([-5.1, 5.1])

    custom_lines = [
        Line2D([0], [0], color='r', lw=1),
        Line2D([0], [0], color='b', lw=1),
    ]
    plt.legend(custom_lines, ['head', 'prediction'])

    def update(t):
        rope_configuration = link_bot_pycommon.make_rope_configuration(*rope_params[t])
        result = constraint_model.evaluate_single(sdf_data, model, threshold, rope_configuration)

        head_scatter.set_offsets(rope_configuration[4:6])

        prediction_scatter.set_offsets(result.predicted_point)

        xs = [rope_configuration[0], rope_configuration[2], rope_configuration[4]]
        ys = [rope_configuration[1], rope_configuration[3], rope_configuration[5]]
        line.set_xdata(xs)
        line.set_ydata(ys)

    fps = 100
    duration_s = 30
    T = rope_params.shape[0]
    interval_ms = int(duration_s * 1000 / T)
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
    contour_spacing_y = 4
    contour_spacing_x = 1
    y_range = np.arange(ymin, ymax + sdf_data.resolution[0], sdf_data.resolution[0] * contour_spacing_y)
    y = y_range[0]
    head_xs_flat = np.arange(xmin, xmax + sdf_data.resolution[1], sdf_data.resolution[1] * contour_spacing_x)
    head_ys_flat = np.ones_like(head_xs_flat) * y

    zeros = np.zeros_like(head_xs_flat)
    rope_configurations = link_bot_pycommon.make_rope_configurations(head_xs_flat, head_ys_flat, zeros, zeros)

    _, predicted_points = model.violated(rope_configurations)

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
        head_xs_flat = np.arange(xmin, xmax + sdf_data.resolution[1], sdf_data.resolution[1] * contour_spacing_x)
        head_ys_flat = np.ones_like(head_xs_flat) * y

        rope_configurations = link_bot_pycommon.make_rope_configurations(head_xs_flat, head_ys_flat, zeros, zeros)

        _, predicted_points = model.violated(rope_configurations)

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

        _, predicted_points = model.violated(rope_configurations)

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


def plot_examples_2(sdf_data, rope_configurations, threshold, model):
    fig = plt.figure()
    binary_image = sdf_data.image > threshold
    plt.imshow(binary_image, extent=sdf_data.extent)

    violated, predicted_points = model.violated(rope_configurations)
    for rope_configuration, predicted_point in zip(rope_configurations, predicted_points):
        plt.plot(rope_configuration[[0, 2, 4]], rope_configuration[[1, 3, 5]], zorder=4)
        plt.plot([rope_configuration[4], predicted_point[0]], [rope_configuration[5], predicted_point[1]], zorder=3, c='k')

    plt.scatter(rope_configurations[:, 4], rope_configurations[:, 5], c='b', s=100, zorder=1)

    pred_color = ['r' if v else 'g' for v in violated]
    plt.scatter(predicted_points[:, 0], predicted_points[:, 1], c=pred_color, s=10, zorder=2)

    return SavableFigure(fig)
