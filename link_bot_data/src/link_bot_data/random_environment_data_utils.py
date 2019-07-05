from __future__ import print_function, division

import os
import sys

import git
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset


def generate_envs(args, full_output_directory, generate_env, save_dict_extras=None):
    if save_dict_extras is None:
        save_dict_extras = {}

    filename_pairs = []
    percentages_positive = []
    constraint_label_types = None
    for i in range(args.envs):
        rope_configurations, labels_dict, sdf_data, percentage_violation = generate_env(args, i)
        constraint_label_types = list(labels_dict.keys())
        percentages_positive.append(percentage_violation)
        if args.outdir:
            rope_data_filename = os.path.join(full_output_directory, 'rope_data_{:d}.npz'.format(i))
            sdf_filename = os.path.join(full_output_directory, 'sdf_data_{:d}.npz'.format(i))

            # FIXME: order matters
            filename_pairs.append([sdf_filename, rope_data_filename])

            save_dict = {
                "rope_configurations": rope_configurations,
            }
            save_dict.update(save_dict_extras)
            for label_type, labels in labels_dict.items():
                save_dict[label_type.name] = labels

            # Save the data
            np.savez(rope_data_filename, **save_dict)
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


def data_directory(outdir, envs, steps):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    full_output_directory = '{}_{}_{}_{}'.format(outdir, sha, envs, steps)
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
