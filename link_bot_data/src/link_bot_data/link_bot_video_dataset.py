import itertools
import os
import re

import tensorflow as tf
from google.protobuf.json_format import MessageToDict

from video_prediction.datasets.video_dataset import VideoDataset


class LinkBotVideoDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super(LinkBotVideoDataset, self).__init__(*args, **kwargs)

        # infer name of image feature
        options = tf.python_io.TFRecordOptions(compression_type=self.hparams.compression_type)
        example = next(tf.python_io.tf_record_iterator(self.filenames[0], options=options))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        image_names = set()
        for name in feature.keys():
            m = re.search('\d+/(\w+)/encoded', name)
            if m:
                image_names.add(m.group(1))
        # look for image_aux1 and image_view0 in that order of priority
        image_name = None
        for name in ['image_aux1', 'image_view0']:
            if name in image_names:
                image_name = name
                break
        if not image_name:
            if len(image_names) == 1:
                image_name = image_names.pop()
            else:
                raise ValueError('The examples have images under more than one name.')
        self.state_like_names_and_shapes['images'] = '%%d/%s/encoded' % image_name, self.hparams.image_shape
        self.state_like_names_and_shapes['rope_configurations'] = '%d/rope_configuration', (self.hparams.rope_config_dim,)
        self.state_like_names_and_shapes['constraints'] = '%d/constraint', (1,)
        self.state_like_names_and_shapes['velocity'] = '%d/1/velocity', (2,)
        self.state_like_names_and_shapes['post_action_velocity'] = '%d/1/post_action_velocity', (2,)
        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = '%d/endeffector_pos', (2,)
            # self.state_like_names_and_shapes['states'] = '%d/rope_configuration', (self.hparams.rope_config_dim,)
        self.action_like_names_and_shapes['actions'] = '%d/action', (2,)
        self.trajectory_constant_names_and_shapes['sdf'] = 'sdf/sdf', [self.hparams.sdf_shape[0], self.hparams.sdf_shape[1], 1]
        self.trajectory_constant_names_and_shapes['sdf_resolution'] = 'sdf/resolution', (2,)
        self.trajectory_constant_names_and_shapes['sdf_origin'] = 'sdf/origin', (2,)
        self._infer_seq_length_and_setup()

    def get_default_hparams_dict(self):
        default_hparams = super(LinkBotVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=12,
            long_sequence_length=30,
            time_shift=2,
            free_space_only=False,
            image_shape=[64, 64, 3],
            sdf_shape=[101, 101],
            rope_config_dim=6,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return False

    def parser(self, serialized_example):
        state_like_seqs, action_like_seqs = super(LinkBotVideoDataset, self).parser(serialized_example)
        return state_like_seqs, action_like_seqs

    def num_examples_per_epoch(self):
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        for filename in self.filenames:
            match = re.search('traj_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count
