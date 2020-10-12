#!/usr/bin/env python
import argparse
import pathlib
from typing import List

import colorama
import numpy as np

import rospy
import state_space_dynamics
from link_bot_data.dynamics_dataset import DynamicsDataset
from moonshine.gpu_config import limit_gpu_mem
from shape_completion_training.model.filepath_tools import load_trial
from shape_completion_training.model_runner import ModelRunner
from state_space_dynamics import train_test

limit_gpu_mem(8)


def train_main(dataset_dirs: List[pathlib.Path],
               checkpoint: pathlib.Path,
               epochs: int,
               ):
    trial_path, params = load_trial(checkpoint.parent.absolute())
    trial_path = trial_path.parent / (trial_path.name + '-observer')
    batch_size = params['batch_size']
    batch_size = 1

    train_dataset = DynamicsDataset(dataset_dirs)
    val_dataset = DynamicsDataset(dataset_dirs)

    model_class = state_space_dynamics.get_model(params['model_class'])
    model = model_class(hparams=params, batch_size=batch_size, scenario=train_dataset.scenario)

    model.encoder.trainable = False
    model.use_observation_feature_loss = True
    model.use_cfm_loss = False

    seed = 0

    runner = ModelRunner(model=model,
                         training=True,
                         params=params,
                         checkpoint=checkpoint,
                         batch_metadata=train_dataset.batch_metadata,
                         trial_path=trial_path)

    train_tf_dataset, val_tf_dataset = train_test.setup_datasets(batch_size, seed, train_dataset, val_dataset)

    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=epochs)

    return trial_path


def main():
    colorama.init(autoreset=True)

    np.set_printoptions(linewidth=250, precision=4, suppress=True, threshold=10000)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches', default=100)
    parser.add_argument('--validation-every', type=int, help='report validation every this many epochs', default=1)
    args = parser.parse_args()

    from time import time

    now = str(int(time()))
    name = f"train_test_{now}"
    rospy.init_node(name)

    train_main(dataset_dirs=args.dataset_dirs,
               checkpoint=args.checkpoint,
               epochs=args.epochs)


if __name__ == '__main__':
    main()
