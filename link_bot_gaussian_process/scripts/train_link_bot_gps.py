#!/usr/bin/env python

import argparse
import os

from colorama import Fore
import gpflow as gpf
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation
from tabulate import tabulate

from link_bot_data.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_data.visualization import plot_rope_configuration
from link_bot_gaussian_process import link_bot_gp, data_reformatting, error_metrics
from link_bot_pycommon import experiments_util
from video_prediction.datasets import dataset_utils


def train(args):
    # Sample a fixed number of transitions
    num_val_examples = 500

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.1))
    gpf.reset_default_session(config=config)
    sess = gpf.get_default_session()
    fwd_model = link_bot_gp.LinkBotGP()

    train_dataset, train_inputs, steps_per_epoch = dataset_utils.get_inputs(args.indir,
                                                                            'link_bot_video',
                                                                            args.dataset_hparams_dict,
                                                                            args.dataset_hparams,
                                                                            mode='train',
                                                                            epochs=1,
                                                                            seed=args.seed,
                                                                            batch_size=args.n_training_examples,
                                                                            shuffle=False)

    val_dataset, val_inputs, _ = dataset_utils.get_inputs(args.indir,
                                                          'link_bot_video',
                                                          args.dataset_hparams_dict,
                                                          args.dataset_hparams,
                                                          mode='val',
                                                          epochs=1,
                                                          seed=args.seed,
                                                          batch_size=num_val_examples)

    try:
        data = sess.run(train_inputs)
    except tf.errors.OutOfRangeError:
        print(Fore.RED + "Dataset does not contain {} examples.".format(args.n_training_examples) + Fore.RESET)
        return

    rope_configurations = data['rope_configurations']
    actions = data['actions']
    # sdfs = data['sdf'][:, 0].squeeze()
    # sdfs = sdfs > 0

    ##########################################################################
    import matplotlib.pyplot as plt
    arrow_width = 0.02
    arena_size = 0.5
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    line = plot_rope_configuration(ax, rope_configurations[idx, 0])
    before = ax.plot(x_0_xs, x_0_ys, color='red', zorder=2)[0]

    arrow = plt.Arrow(x_0[4], x_0[5], actions[0, 0, 0], actions[0, 0, 1], width=arrow_width, zorder=3)
    patch = ax.add_patch(arrow)

    ax.set_title("0")

    x_0 = rope_configurations[0, 1]
    x_0_xs = [x_0[0], x_0[2], x_0[4]]
    x_0_ys = [x_0[1], x_0[3], x_0[5]]

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-arena_size, arena_size])
    plt.ylim([-arena_size, arena_size])

    def update(i):
        nonlocal patch

        x_i = rope_configurations[traj_idx, t]
        x_i_xs = [x_i[0], x_i[2], x_i[4]]
        x_i_ys = [x_i[1], x_i[3], x_i[5]]
        before.set_xdata(x_i_xs)
        before.set_ydata(x_i_ys)

        patch.remove()
        arrow = plt.Arrow(x_i[4], x_i[5], actions[traj_idx, t, 0], actions[traj_idx, t, 1], width=arrow_width, zorder=3)
        patch = ax.add_patch(arrow)

        ax.set_title("{} {} ".format(traj_idx, t))

    anim = FuncAnimation(fig, update, frames=rope_configurations.shape[0] * rope_configurations.shape[1], interval=1000, repeat=False)
    plt.show()
    return
    ##########################################################################

    fwd_train_data = data_reformatting.format_forward_data_gz_tfrecords(rope_configurations, actions)
    fwd_train_x = fwd_train_data[2]
    fwd_train_y = fwd_train_data[1]

    fwd_val_data = data_reformatting.format_forward_data_gz_tfrecords(rope_configurations, actions)
    fwd_val_x = fwd_val_data[2]
    fwd_val_y = fwd_val_data[1]

    import matplotlib.pyplot as plt
    anim = link_bot_gp.animate_training_data(rope_configurations, actions, sdfs, arena_size=0.5, interval=2000)
    plt.show()

    # Train
    ###########################################################################

    print("Training forward model")
    fwd_model.train(fwd_train_x, fwd_train_y, verbose=args.verbose, maximum_training_iterations=args.max_iters,
                    n_inducing_points=args.n_inducing_points)

    # Save
    ###########################################################################
    if not args.dont_save:
        log_path = experiments_util.experiment_name('separate_independent', 'gpf')
        fwd_model.save(log_path, 'fwd_model')

    evaluate(args, fwd_model, fwd_val_x, fwd_val_y)


def eval(args):
    fwd_model = link_bot_gp.LinkBotGP()

    fwd_model_path = os.path.join(args.model_dir, "fwd_model")

    fwd_model.load(fwd_model_path)

    test_dataset = MultiEnvironmentDataset.load_dataset(args.test_dataset)

    fwd_test_data = data_reformatting.format_forward_data_gz(args, test_dataset)
    fwd_test_x = fwd_test_data[3]
    fwd_test_y = fwd_test_data[1]

    evaluate(args, fwd_model, fwd_test_x, fwd_test_y)


def evaluate(args, fwd_model, fwd_test_x, fwd_test_y):
    headers = ['error metric', 'min', 'max', 'mean', 'median', 'std']
    aggregate_metrics = error_metrics.fwd_model_error_metrics(fwd_model, fwd_test_x, fwd_test_y)
    table = tabulate(aggregate_metrics, headers=headers, tablefmt='github', floatfmt='6.3f')
    print(table)
    with open("metrics.md", 'w') as f:
        f.writelines(table)
        f.write("\n")


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=1000)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('indir')
    train_parser.add_argument('dataset_hparams_dict')
    train_parser.add_argument('--dataset-hparams')
    train_parser.add_argument('--seed', type=int, default=0)
    train_parser.add_argument('--n-training-examples', type=int, default=1000)
    train_parser.add_argument('--max-iters', type=int, default=200)
    train_parser.add_argument('--n-inducing-points', type=int, default=20)
    train_parser.add_argument('--verbose', action='store_true')
    train_parser.add_argument('--log')
    train_parser.add_argument('--dont-save', action='store_true')
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('indir')
    eval_parser.add_argument('dataset_hparams_dict')
    eval_parser.add_argument('--dataset-hparams')
    eval_parser.add_argument('model_dir')
    eval_parser.add_argument('--seed', type=int, default=0)
    eval_parser.add_argument('--verbose', action='store_true')
    eval_parser.set_defaults(func=eval)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    args.func(args)


if __name__ == '__main__':
    main()
