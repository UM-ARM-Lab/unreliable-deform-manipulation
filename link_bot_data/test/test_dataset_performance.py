#!/usr/bin/env python
import argparse
import pathlib
import time

import progressbar
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import balance
from moonshine.image_functions import add_traj_image, add_transition_image
from link_bot_planning.get_scenario import get_scenario
from link_bot_pycommon.args import my_formatter

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='val')
    parser.add_argument('--n-repetitions', type=int, default=1)

    args = parser.parse_args()

    # dataset = DynamicsDataset(args.dataset_dir)
    params = {
        "pre_close_threshold": 0.1,
        "post_close_threshold": 0.1,
        "discard_pre_far": True,
        "balance": True,
        "state_key": "link_bot",
    }
    dataset = ClassifierDataset(args.dataset_dir, params)

    batch_size = 64

    t0 = time.perf_counter()
    tf_dataset = dataset.get_datasets(mode=args.mode)

    # tf_dataset = add_traj_image(tf_dataset)
    tf_dataset = add_transition_image(tf_dataset,
                                      states_keys=["link_bot"],
                                      scenario=get_scenario("link_bot"),
                                      local_env_h=50,
                                      local_env_w=50,
                                      rope_image_k=1000)
    # tf_dataset = balance(tf_dataset, label_key='label')
    # tf_dataset = tf_dataset.shuffle(1024)
    tf_dataset = tf_dataset.batch(batch_size)

    time_to_load = time.perf_counter() - t0
    print("Time to Load (s): {:5.3f}".format(time_to_load))

    try:
        for _ in range(args.n_repetitions):
            for e in progressbar.progressbar(tf_dataset):
                pass
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
