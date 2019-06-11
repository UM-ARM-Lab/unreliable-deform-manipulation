import os
import numpy as np
from colorama import Fore
import json
from link_bot_pycommon.link_bot_pycommon import SDF


class Environment:

    def __init__(self, sdf_data, rope_data):
        self.sdf_data = sdf_data
        self.rope_data = rope_data


class MultiEnvironmentDataset:

    def __init__(self, filename_pairs):
        self.abs_filename_pairs = []
        for filename_pair in filename_pairs:
            abs_filename_pair = []
            for filename in filename_pair:
                if 'LINK_BOT_ROOT' in os.environ:
                    link_bot_root = os.environ['LINK_BOT_ROOT']
                else:
                    link_bot_root = os.path.expanduser("~/catkin_ws/src/link_bot")
                    print(Fore.YELLOW + "env var LINK_BOT_ROOT not defined, assuming {}".format(link_bot_root) + Fore.RESET)

                abs_filename = os.path.join(link_bot_root)
                abs_filename_pair.append(abs_filename)
            self.abs_filename_pairs.append(abs_filename_pair)

        # make sure all the SDFs have the same shape
        self.sdf_shape = None
        self.n_environments = len(filename_pairs)
        self.environments = np.ndarray([self.n_environments], dtype=np.object)
        for i, (sdf_filename, rope_data_filename) in enumerate(filename_pairs):
            sdf_data = SDF.load(sdf_filename)
            rope_data = np.load(rope_data_filename)

            if self.sdf_shape is None:
                self.sdf_shape = sdf_data.sdf.shape
            error_msg = "SDFs shape {} doesn't match shape {}".format(self.sdf_shape, sdf_data.sdf.shape)
            assert self.sdf_shape == sdf_data.sdf.shape, error_msg

            env = Environment(sdf_data, rope_data)
            self.environments[i] = env

    @staticmethod
    def load_dataset(dataset_filename):
        dataset_dict = json.load(open(dataset_filename, 'r'))
        filename_pairs = dataset_dict['filename_pairs']
        dataset = MultiEnvironmentDataset(filename_pairs)
        return dataset

    def save(self, dataset_filename):
        dataset_dict = {
            'filename_pairs': self.abs_filename_pairs
        }
        json.dump(dataset_dict, open(dataset_filename, 'w'), sort_keys=True, indent=4)
