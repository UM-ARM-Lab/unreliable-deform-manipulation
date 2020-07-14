#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import pathlib
import rospy

from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from visualization_msgs.msg import MarkerArray, Marker
from link_bot_data.visualization import rviz_arrow
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_data.recovery_dataset import RecoveryDataset


limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')

    args = parser.parse_args()

    rospy.init_node('vis_recovery_dataset')

    dataset = RecoveryDataset(args.dataset_dirs)

    visualize_dataset(args, dataset)


def visualize_dataset(args, dataset):
    tf_dataset = dataset.get_datasets(mode=args.mode)

    scenario = get_scenario(dataset.hparams['scenario'])
    testing_pub = rospy.Publisher("testing", MarkerArray, queue_size=10, latch=True)

    idx = 0
    deltas = []
    for example_idx, example in enumerate(tf_dataset):
        p_accepts = tf.reduce_mean(example['accept_probabilities'], axis=1)
        max_scores = tf.reduce_max(example['accept_probabilities'], axis=1)
        anim = RvizAnimationController(np.arange(dataset.horizon))
        scenario.plot_environment_rviz(example)
        p_accept = tf.reduce_max(p_accepts)
        print(p_accept)

        while not anim.done:

            t = anim.t()
            mean_score_t = p_accepts[t]
            max_score_t = max_scores[t]
            s_t = {k: example[k][t] for k in dataset.state_keys}
            if t < dataset.horizon - 1:
                a_t = {k: example[k][t] for k in dataset.action_keys}

                delta1 = a_t['gripper1_position'] - s_t['gripper1']
                delta2 = a_t['gripper2_position'] - s_t['gripper2']
                deltas.append(delta1)
                deltas.append(delta2)
                mean_delta = tf.reduce_mean(deltas, axis=0)

                scenario.plot_action_rviz(s_t, a_t, label='observed')

                marker_msg = MarkerArray()
                marker = Marker()
                marker.scale.x = 0.005
                marker.scale.y = 0.005
                marker.scale.z = 0.005
                marker.action = Marker.ADD
                marker.type = Marker.SPHERE
                marker.header.frame_id = "/world"
                marker.header.stamp = rospy.Time.now()
                marker.ns = 'action'
                marker.id = idx
                marker.color.r = 0.9
                marker.color.g = 0.9
                marker.color.b = 0.3
                marker.color.a = 1.0
                marker.pose.position.x = delta1[0]
                marker.pose.position.y = delta1[1]
                marker.pose.position.z = delta1[2]
                marker.pose.orientation.w = 1

                mean_arrow = Marker()
                mean_arrow.scale.x = 0.005
                mean_arrow.scale.y = 0.005
                mean_arrow.scale.z = 0.005
                mean_arrow.action = Marker.ADD
                mean_arrow.type = Marker.SPHERE
                mean_arrow.header.frame_id = "/world"
                mean_arrow.header.stamp = rospy.Time.now()
                mean_arrow.ns = 'action'
                mean_arrow.id = idx
                mean_arrow.color.r = 0.9
                mean_arrow.color.g = 0.9
                mean_arrow.color.b = 0.3
                mean_arrow.color.a = 0.2
                mean_arrow.pose.position.x = delta1[0]
                mean_arrow.pose.position.y = delta1[1]
                mean_arrow.pose.position.z = delta1[2]
                mean_arrow.pose.orientation.w = 1

                marker_msg.markers.append(marker)
                marker_msg.markers.append(rviz_arrow([0, 0, 0], mean_delta, 0, 1, 0, 1, idx=0, label='observed'))
                testing_pub.publish(marker_msg)
            scenario.plot_state_rviz(s_t, label='observed', color=cm.Reds(mean_score_t))
            anim.step()

            idx += 1


if __name__ == '__main__':
    main()
