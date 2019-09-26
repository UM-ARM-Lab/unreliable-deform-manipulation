import pathlib
import numpy as np
import random

import tensorflow as tf

import json

from google.protobuf.json_format import MessageToDict

from link_bot_data.video_prediction_dataset_utils import float_feature


class ClassifierDataset:

    def __init__(self,
                 dataset_dir: pathlib.Path,
                 ):

        self.dataset_dir = dataset_dir
        dataset_hparams_filename = dataset_dir / 'hparams.json'
        self.hparams = json.load(open(str(dataset_hparams_filename), 'r'))

    def parser(self, sdf_shape, n_state, n_action):
        def _parser(serialized_example):
            features = {
                'sdf/sdf': tf.FixedLenFeature(sdf_shape, tf.float32),
                'sdf/extent': tf.FixedLenFeature([4], tf.float32),
                'sdf/res': tf.FixedLenFeature([1], tf.float32),
                'sdf/origin': tf.FixedLenFeature([2], tf.float32),
                'sdf/w': tf.FixedLenFeature([1], tf.float32),
                'sdf/h': tf.FixedLenFeature([1], tf.float32),
                'state': tf.FixedLenFeature([n_state], tf.float32),
                'next_state': tf.FixedLenFeature([n_state], tf.float32),
                'action': tf.FixedLenFeature([n_action], tf.float32),
                'planned_state': tf.FixedLenFeature([n_state], tf.float32),
                'planned_next_state': tf.FixedLenFeature([n_state], tf.float32),
            }
            features = tf.parse_single_example(serialized_example, features=features)

            # Convert the flattened SDF tensor into a (H,W,1) tensor

            return features

        return _parser

    @classmethod
    def make_features_dict(cls,
                           actual_local_sdf,
                           actual_local_sdf_extent,
                           actual_local_sdf_origin,
                           planned_local_sdf,
                           planned_local_sdf_extent,
                           planned_local_sdf_origin,
                           sdf_h,
                           sdf_w,
                           res,
                           state,
                           next_state,
                           action,
                           planned_state,
                           planned_next_state):
        features = {
            'actual_sdf/sdf': float_feature(actual_local_sdf.flatten()),
            'actual_sdf/extent': float_feature(np.array(actual_local_sdf_extent)),
            'actual_sdf/origin': float_feature(np.array(actual_local_sdf_origin)),
            'planned_sdf/sdf': float_feature(planned_local_sdf.flatten()),
            'planned_sdf/extent': float_feature(np.array(planned_local_sdf_extent)),
            'planned_sdf/origin': float_feature(np.array(planned_local_sdf_origin)),
            'res': float_feature(np.array([res])),
            'w': float_feature(np.array([sdf_h])),
            'h': float_feature(np.array([sdf_w])),
            'state': float_feature(state),
            'next_state': float_feature(next_state),
            'action': float_feature(action),
            'planned_state': float_feature(planned_state),
            'planned_next_state': float_feature(planned_next_state),
        }
        return features

    @classmethod
    def make_serialized_example(cls, *args):
        features = cls.make_features_dict(*args)
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        example = example_proto.SerializeToString()
        return example

    def get_dataset(self,
                    mode: str,
                    num_epochs: int,
                    compression_type: str,
                    shuffle: bool = True,
                    seed: int = 0,
                    ):

        filenames = [str(filename) for filename in self.dataset_dir.glob("{}/*.tfrecords".format(mode))]

        options = tf.python_io.TFRecordOptions(compression_type=compression_type)
        example = next(tf.python_io.tf_record_iterator(filenames[3], options=options))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        for feature_name, feature_description in feature.items():
            if feature_name == 'sdf/sdf':
                flattened_sdf_shape = len(feature_description['floatList']['value'])
                s = int(np.sqrt(flattened_sdf_shape))
                assert s * s == flattened_sdf_shape, "flat SDF was not square, had size {}, sqrt {}".format(flattened_sdf_shape,
                                                                                                            s)
                print(flattened_sdf_shape, s)
                sdf_shape = [s, s]
            if feature_name == 'state':
                n_state = len(feature_description['floatList']['value'])
            if feature_name == 'action':
                n_action = len(feature_description['floatList']['value'])

        if shuffle:
            random.shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024, compression_type=compression_type)

        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=num_epochs))
        else:
            dataset = dataset.repeat(num_epochs)

        dataset = dataset.map(self.parser(sdf_shape, n_state, n_action))

        return dataset
