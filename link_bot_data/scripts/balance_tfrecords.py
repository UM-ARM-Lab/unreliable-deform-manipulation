#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import glob
import os
import pathlib
import re

import numpy as np
from link_bot_data import video_prediction_dataset_utils
import tensorflow as tf
from google.protobuf.json_format import MessageToDict

tf.enable_eager_execution()


def write_balanced_dataset(mode, args):
    # look for tfrecords in input_dir and input_dir/mode directories
    indir = os.path.join(args.indir, mode)
    filenames = glob.glob(os.path.join(indir, '*.tfrecord*'))
    filenames = sorted(filenames)
    options = tf.python_io.TFRecordOptions(compression_type=args.compression_type)

    constraint_feature_pattern = re.compile("(\d+)/constraint")
    any_time_based_feature_pattern = re.compile("(\d+)/(.+)")
    balanced_trajectories = []
    for i, filename in enumerate(filenames):
        for j, traj in enumerate(tf.python_io.tf_record_iterator(filename, options=options)):
            traj_message = MessageToDict(tf.train.Example.FromString(traj))
            traj_dict = traj_message['features']['feature']

            time_based_features = np.array([{} for _ in range(100)], dtype=dict)
            non_time_based_features = {}
            for key, value in traj_dict.items():
                match = re.fullmatch(any_time_based_feature_pattern, key)
                if match:
                    t = int(match.group(1))
                    time_based_features[t][key] = value
                else:
                    non_time_based_features[key] = value

            positive_time_based_features = []
            negative_time_based_features = []
            for t, features_at_time_t in enumerate(time_based_features):
                constraint = int(features_at_time_t['{:d}/constraint'.format(t)]['floatList']['value'][0])
                if constraint:
                    positive_time_based_features.append(features_at_time_t)
                else:
                    negative_time_based_features.append(features_at_time_t)

            balanced_time_based_features = list(zip(positive_time_based_features, negative_time_based_features))
            if len(balanced_time_based_features) == 0:
                print("Traj {} in file {} cannot be balanced - skipping".format(j, filename))
                continue

            balanced_time_based_features_dict = {}
            for pos_ex, neg_ex in balanced_time_based_features:
                balanced_time_based_features_dict.update(pos_ex)
                balanced_time_based_features_dict.update(neg_ex)

            non_time_based_features.update(balanced_time_based_features_dict)
            balanced_trajectory = non_time_based_features
            balanced_trajectories.append(balanced_trajectory)

    for i in range(0, len(balanced_trajectories), args.trajs_per_record):
        trajs = balanced_trajectories[i:i + args.trajs_per_record]
        traj_idx_start = i * args.trajs_per_record
        traj_idx_end = traj_idx_start + args.trajs_per_record - 1

        outdir = pathlib.Path(args.outdir)
        filename = outdir / mode / 'traj_{}_to_{}.tfrecord'.format(traj_idx_start, traj_idx_end)
        with tf.python_io.TFRecordWriter(str(filename), options=options) as writer:
            for traj in trajs:
                dict_of_feature = {}
                for feature_name, feature_desc in traj.items():
                    match = re.fullmatch(any_time_based_feature_pattern, feature_name)
                    if match:
                        new_feature_name = '{:d}/{}'.format(k, match.group(2))
                        print(feature_name, new_feature_name)
                    feature_type = list(feature_desc.keys())[0]
                    feature_value = list(feature_desc.values())[0]['value']
                    if feature_type == 'floatList':
                        feature_value_np = np.array(feature_value)
                        feature_proto = video_prediction_dataset_utils.float_feature(feature_value_np)
                        dict_of_feature[feature_name] = feature_proto
                    elif feature_type == 'bytesList':
                        feature_proto = video_prediction_dataset_utils.bytes_feature(bytes(feature_value[0], encoding='utf-8'))
                        dict_of_feature[feature_name] = feature_proto
                ex_features = tf.train.Features(feature=dict_of_feature)
                example_proto = tf.train.Example(features=ex_features)
                example_str = example_proto.SerializeToString()
                writer.write(example_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir')
    parser.add_argument('outdir')
    parser.add_argument('--trajs-per-record', type=int, default=512)
    parser.add_argument('--compression-type', default='ZLIB', choices=['ZLIB', 'GZIP', ''])

    args = parser.parse_args()

    write_balanced_dataset('test', args)


if __name__ == '__main__':
    main()
