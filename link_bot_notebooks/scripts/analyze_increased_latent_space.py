#!/usr/bin/env python

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from link_bot_notebooks import linear_tf_model as m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoints", help="set of checkpoints to evaluate", nargs="+")
    parser.add_argument("--training_dataset", help='dataset to test on',
                        default="../link_bot_teleop/data/250_50_random3.npy")
    parser.add_argument("--testing_dataset", help='dataset to test on',
                        default="../link_bot_teleop/data/50_50_random_test.npy")
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    args = parser.parse_args()

    if len(args.model_checkpoints) < 1:
        print("You must supply 1 or more model checkpoints")
        return

    def evaluate(dataset, checkpoint):
        goal = np.zeros((1, args.N))
        log_data = np.load(dataset)
        x = log_data[:, :, :]
        dt = x[0, 1, 0] - x[0, 0, 0]
        model_args = {
            'checkpoint': checkpoint,
            'log': False,
            'debug': False,
        }
        tf.reset_default_graph()
        print(checkpoint, dataset)
        model = m.LinearTFModel(model_args, x.shape[0], args.N, args.M, args.L, dt, x.shape[1] - 1)
        model.load()
        loss = model.evaluate(x, goal, display=False)[-1]
        print(loss)
        return loss

    train_losses = []
    test_losses = []
    # Handle case of when model checkpoints
    if not isinstance(args.model_checkpoints, list):
        args.model_checkpoints = [args.model_checkpoints]
    for ckpt in args.model_checkpoints:
        train_losses.append(evaluate(args.training_dataset, ckpt))
        test_losses.append(evaluate(args.testing_dataset, ckpt))

    x = np.arange(len(train_losses))
    plt.bar(x * 2, train_losses, label="training")
    plt.bar(x * 2 + 1, test_losses, label="testing")
    plt.ylabel("loss")
    plt.xticks(x * 2 + 1 / 2, args.model_checkpoints)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
