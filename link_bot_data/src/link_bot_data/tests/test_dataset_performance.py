#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import progressbar

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.image_functions import add_traj_image_to_example
from moonshine.old_image_functions import add_traj_image_to_example as add_traj_image_to_example_old


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')
    parser.add_argument('--n-repetitions', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)

    args = parser.parse_args()

    dataset = ClassifierDataset(args.dataset_dir)
    dataset.cache_negative = False

    t0 = perf_counter()
    tf_dataset = dataset.get_datasets(mode=args.mode)
    scenario = get_scenario('link_bot')
    tf_dataset = tf_dataset.batch(args.batch_size)

    time_to_load = perf_counter() - t0
    print("Time to Load (s): {:5.3f}".format(time_to_load))

    n = 8192
    batches = int(n / args.batch_size)
    try:
        # ram_usage = []
        for _ in range(args.n_repetitions):
            # None
            t0 = perf_counter()
            for e in progressbar.progressbar(tf_dataset.take(batches)):
                pass
            print('{:.5f}'.format(perf_counter() - t0))

            # NEW
            t0 = perf_counter()
            for e in progressbar.progressbar(tf_dataset.take(batches)):
                e = add_traj_image_to_example(scenario=scenario,
                                              example=e,
                                              local_env_w=100,
                                              states_keys=['link_bot'],
                                              local_env_h=100,
                                              rope_image_k=10000,
                                              batch_size=args.batch_size)
            print('{:.5f}'.format(perf_counter() - t0))

            # OLD
            t0 = perf_counter()
            for e in progressbar.progressbar(tf_dataset.take(batches)):
                e = add_traj_image_to_example_old(scenario=scenario,
                                                  example=e,
                                                  local_env_w=100,
                                                  states_keys=['link_bot'],
                                                  local_env_h=100,
                                                  rope_image_k=10000,
                                                  batch_size=args.batch_size)
            print('{:.5f}'.format(perf_counter() - t0))

        # plt.plot(ram_usage)
        # plt.xlabel("iteration")
        # plt.ylabel("ram usage (bytes)")
        # plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
