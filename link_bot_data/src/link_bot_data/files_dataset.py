import pathlib
import shutil

import numpy as np

from link_bot_data.base_dataset import DEFAULT_TEST_SPLIT, DEFAULT_VAL_SPLIT


class FilesDataset:

    def __init__(self, root_dir: pathlib.Path):
        self.root_dir = root_dir
        self.paths = []

    def add(self, full_filename: pathlib.Path):
        self.paths.append(full_filename)

    def split(self, shuffle=False):
        (self.root_dir / 'train').mkdir(exist_ok=True)
        (self.root_dir / 'test').mkdir(exist_ok=True)
        (self.root_dir / 'val').mkdir(exist_ok=True)

        n_files = len(self.paths)
        n_validation = int(DEFAULT_VAL_SPLIT * n_files)
        n_testing = int(DEFAULT_TEST_SPLIT * n_files)
        if shuffle:
            rng = np.random.RandomState(0)
            rng.shuffle(self.paths)
        val_files = self.paths[0:n_validation]
        self.paths = self.paths[n_validation:]
        if shuffle:
            rng.shuffle(self.paths)
        test_files = self.paths[0:n_testing]
        train_files = self.paths[n_testing:]

        for train_file in train_files:
            out = self.root_dir / 'train' / train_file.name
            shutil.move(train_file, out)

        for test_file in test_files:
            out = self.root_dir / 'test' / test_file.name
            shutil.move(test_file, out)

        for val_file in val_files:
            out = self.root_dir / 'val' / val_file.name
            shutil.move(val_file, out)
