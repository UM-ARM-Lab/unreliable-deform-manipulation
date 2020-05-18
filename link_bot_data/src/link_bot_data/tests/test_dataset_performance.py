#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import progressbar

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_pycommon.args import my_formatter


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')
    parser.add_argument('--n-repetitions', type=int, default=1)

    args = parser.parse_args()

    # dataset = DynamicsDataset(args.dataset_dir)
    dataset = ClassifierDataset(args.dataset_dir)

    batch_size = 64

    t0 = perf_counter()
    tf_dataset = dataset.get_datasets(mode=args.mode)
    tf_dataset = tf_dataset.batch(batch_size)

    # tf_dataset = add_transition_image(tf_dataset,
    #                                   states_keys=["link_bot"],
    #                                   scenario=get_scenario("link_bot"),
    #                                   local_env_h=50,
    #                                   local_env_w=50,
    #                                   rope_image_k=1000)
    tf_dataset = tf_dataset.shuffle(512)

    time_to_load = perf_counter() - t0
    print("Time to Load (s): {:5.3f}".format(time_to_load))

    # tf_dataset = tf_dataset.cache(cachename())
    n_positive = 0
    try:
        for _ in range(args.n_repetitions):
            t0 = perf_counter()
            for e in progressbar.progressbar(tf_dataset):
                # print('{:.5f}'.format(perf_counter() - t0))
                pass
            print('{:.5f}'.format(perf_counter() - t0))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
