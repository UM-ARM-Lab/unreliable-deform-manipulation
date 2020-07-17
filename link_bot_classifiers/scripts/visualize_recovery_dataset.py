#!/usr/bin/env python
import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import pathlib
import rospy

from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import listify
from moonshine.gpu_config import limit_gpu_mem
from visualization_msgs.msg import MarkerArray, Marker
from link_bot_data.visualization import rviz_arrow
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_data.recovery_dataset import RecoveryDataset


limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=5)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')

    args = parser.parse_args()

    rospy.init_node('vis_recovery_dataset')

    dataset = RecoveryDataset(args.dataset_dirs)

    visualize_dataset(args, dataset)


def visualize_dataset(args, dataset: RecoveryDataset):
    tf_dataset = dataset.get_datasets(mode=args.mode)

    scenario = get_scenario(dataset.hparams['scenario'])

    idx = 0
    weighted_action = None
    out_examples = []
    for example_idx, example in enumerate(tf_dataset):
        n_accepts = tf.math.count_nonzero(example['accept_probabilities'][1] > 0.5)
        score = n_accepts / dataset.n_action_samples

        out_example = example
        out_example['score'] = score

        out_examples.append(out_example)

    del example
    out_examples = sorted(out_examples, key=lambda out_example: out_example['score'], reverse=True)

    for out_example in out_examples:
        anim = RvizAnimationController(np.arange(dataset.horizon))
        scenario.plot_environment_rviz(out_example)
        # score = out_example['score'].numpy()
        print(score)
        while not anim.done:

            t = anim.t()
            state = {k: out_example[k][0] for k in dataset.state_keys}
            action = {k: out_example[k][0] for k in dataset.action_keys}

            local_action = scenario.put_action_local_frame(state, action)
            s_t = {k: out_example[k][t] for k in dataset.state_keys}
            if t < dataset.horizon - 1:
                a_t = {k: out_example[k][t] for k in dataset.action_keys}

                delta1 = a_t['gripper1_position'] - s_t['gripper1']

                scenario.plot_action_rviz(s_t, a_t, label='observed')
            scenario.plot_state_rviz(s_t, label='observed', color=cm.Reds(score))
            anim.step()
            idx += 1


if __name__ == '__main__':
    main()
