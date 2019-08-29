from __future__ import print_function, division

import os
import sys

import git
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

from ignition import markers


def publish_marker(args, target_x, target_y, marker_size=0.01):
    target_marker = markers.make_marker(rgb=[1, 0, 0], id=1, scale=marker_size)
    target_marker.pose.position.x = target_x
    target_marker.pose.position.y = target_y
    markers.publish(target_marker)


def publish_markers(args, target_x, target_y, rope_x, rope_y, marker_size=0.01):
    target_marker = markers.make_marker(rgb=[1, 0, 0], id=1, scale=marker_size)
    target_marker.pose.position.x = target_x
    target_marker.pose.position.y = target_y
    rope_marker = markers.make_marker(rgb=[0, 1, 0], id=2, scale=marker_size)
    rope_marker.pose.position.x = rope_x
    rope_marker.pose.position.y = rope_y
    markers.publish(target_marker)
    markers.publish(rope_marker)


def generate_envs(args, full_output_directory, generate_env, save_dict_extras=None):
    from link_bot_data.multi_environment_datasets import MultiEnvironmentDataset

    if save_dict_extras is None:
        save_dict_extras = {}

    filename_pairs = []
    percentages_positive = []
    constraint_label_types = None
    for i in range(args.envs):
        data_dict, labels_dict, sdf_data, percentage_violation = generate_env(args, i)
        constraint_label_types = list(labels_dict.keys())
        percentages_positive.append(percentage_violation)
        if args.outdir:
            rope_data_filename = os.path.join(full_output_directory, 'rope_data_{:d}.npz'.format(i))
            sdf_filename = os.path.join(full_output_directory, 'sdf_data_{:d}.npz'.format(i))

            # FIXME: order matters
            filename_pairs.append([sdf_filename, rope_data_filename])

            data_dict.update(labels_dict)
            data_dict.update(save_dict_extras)

            # Save the data
            np.savez(rope_data_filename, **data_dict)
            sdf_data.save(sdf_filename)

        print(".", end='')
        sys.stdout.flush()

    print("done")

    mean_percentage_positive = np.mean(percentages_positive)
    print("Class balance: mean % positive: {}".format(mean_percentage_positive))

    if args.outdir:
        dataset_filename = os.path.join(full_output_directory, 'dataset.json')
        dataset = MultiEnvironmentDataset(filename_pairs, constraint_label_types=constraint_label_types,
                                          n_obstacles=args.n_obstacles, obstacle_size=args.obstacle_size, seed=args.seed)
        dataset.save(dataset_filename)


def data_directory(outdir, *names):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    format_string = "{}_{}_" + "{}_" * (len(names) - 1) + "{}"
    full_output_directory = format_string.format(outdir, sha, *names)
    if outdir:
        if os.path.isfile(full_output_directory):
            print(Fore.RED + "argument outdir is an existing file, aborting." + Fore.RESET)
            return
        elif not os.path.isdir(full_output_directory):
            os.mkdir(full_output_directory)
    return full_output_directory


def plot_sdf_and_ovs(args, sdf_data, threshold, rope_configuration, sdf_constraint_labels=None, ovs_constraint_labels=None,
                     combined_labels=None):
    plt.figure()
    binary = sdf_data.sdf < threshold
    plt.imshow(np.flipud(binary.T), extent=sdf_data.extent)

    xs = [rope_configuration[0], rope_configuration[2], rope_configuration[4]]
    ys = [rope_configuration[1], rope_configuration[3], rope_configuration[5]]
    if sdf_constraint_labels is not None:
        sdf_constraint_color = 'r' if sdf_constraint_labels else 'g'
    else:
        sdf_constraint_color = 'c'

    if ovs_constraint_labels is not None:
        overstretched_constraint_color = 'r' if ovs_constraint_labels else 'g'
    else:
        overstretched_constraint_color = 'c'

    if combined_labels is not None:
        overstretched_constraint_color = 'r' if combined_labels else 'g'
        sdf_constraint_color = 'r' if combined_labels else 'g'

    plt.plot(xs, ys, linewidth=0.5, zorder=1, c=overstretched_constraint_color)
    plt.scatter(rope_configuration[4], rope_configuration[5], s=16, c=sdf_constraint_color, zorder=2)

    if args.show_sdf_data:
        plt.figure()
        plt.imshow(np.flipud(sdf_data.sdf.T), extent=sdf_data.extent)
        subsample = 2
        x_range = np.arange(sdf_data.extent[0], sdf_data.extent[1], subsample * sdf_data.resolution[0])
        y_range = np.arange(sdf_data.extent[0], sdf_data.extent[1], subsample * sdf_data.resolution[1])
        y, x = np.meshgrid(y_range, x_range)
        dx = sdf_data.gradient[::subsample, ::subsample, 0]
        dy = sdf_data.gradient[::subsample, ::subsample, 1]
        plt.quiver(x, y, dx, dy, units='x', scale=10)


