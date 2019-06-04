#!/usr/bin/env python
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from link_bot_notebooks import constraint_model as m
from link_bot_notebooks import toy_problem_optimization_common as tpoc

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0


def evaluate(sdf, sdf_resolution, sdf_origin, model, threshold, rope_configuration, plot=False):
    global true_positives, true_negatives, false_positives, false_negatives

    rope_configuration = np.atleast_2d(rope_configuration)
    violated, pt = model.violated(rope_configuration)
    row_col = tpoc.point_to_sdf_idx(rope_configuration[0, 4], rope_configuration[0, 5], sdf_resolution, sdf_origin)
    true_violated = sdf[row_col] < threshold

    if violated:
        if true_violated:
            true_positives += 1
        else:
            false_positives += 1
    else:
        if true_violated:
            true_negatives += 1
        else:
            false_negatives += 1

    if plot:
        plt.figure()
        img = Image.fromarray(np.uint8(np.flipud(sdf.T) > threshold))
        small_sdf = img.resize((50, 50))
        plt.imshow(small_sdf, extent=[-5, 5, -5, 5])
        plt.plot(rope_configuration[[0, 0, 0], [0, 2, 4]], rope_configuration[[0, 0, 0], [1, 3, 5]], label='rope')
        if violated:
            pred_color = 'r'
        else:
            pred_color = 'g'
        if true_violated:
            true_color = 'r'
        else:
            true_color = 'g'
        plt.scatter(pt[0, 0], pt[0, 1], s=50, c=pred_color, label='pred')
        plt.scatter(rope_configuration[0, 4], rope_configuration[0, 5], s=10, c=true_color, label='true')
        plt.legend()


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=m.ConstraintModelType.strings())
    parser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    parser.add_argument("checkpoint", help="eval the *.ckpt name")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", help='use this dataset instead of random rope configurations')

    args = parser.parse_args()

    sdf, sdf_gradient, sdf_resolution, sdf_origin = tpoc.load_sdf(args.sdf)
    args_dict = vars(args)
    model = m.ConstraintModel(args_dict, sdf, sdf_gradient, sdf_resolution, sdf_origin, args.N)
    model.setup()

    threshold = 0.2

    if args.dataset:
        data = np.load(args.dataset)
        for i, traj in enumerate(data['states']):
            for t, rope_configuration in enumerate(traj):
                plot = False
                if i < 5 and t == 0:
                    plot = True
                evaluate(sdf, sdf_resolution, sdf_origin, model, threshold, rope_configuration, plot=plot)
    else:
        for i in range(1000):
            plot = False
            if i < 5:
                plot = True

            rope_configuration = np.zeros((1, 6))
            rope_configuration[0, 0] = np.random.uniform(-3, 3)
            rope_configuration[0, 1] = np.random.uniform(-3, 3)
            theta1 = np.random.uniform(-np.pi, np.pi)
            theta2 = np.random.uniform(-np.pi, np.pi)
            rope_configuration[0, 2] = rope_configuration[0, 0] + np.cos(theta1)
            rope_configuration[0, 3] = rope_configuration[0, 1] + np.sin(theta1)
            rope_configuration[0, 4] = rope_configuration[0, 2] + np.cos(theta2)
            rope_configuration[0, 5] = rope_configuration[0, 3] + np.sin(theta2)
            evaluate(sdf, sdf_resolution, sdf_origin, model, threshold, rope_configuration, plot=plot)

    print('accuracy:',
          (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives))
    print('precision:', true_positives / (true_positives + false_positives))
    print('recall:', true_negatives / (true_negatives + false_negatives))

    plt.show()


if __name__ == '__main__':
    main()
