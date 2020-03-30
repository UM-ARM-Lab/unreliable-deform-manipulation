#!/usr/bin/env python

import argparse
import json
import os
import pathlib
import re
import shutil

from colorama import Fore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indirs", nargs="*", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()

    if not args.dry_run:
        path = args.indirs[0] / 'hparams.json'
        new_path = args.outdir / 'hparams.json'
        # log this operation in the params!
        hparams = json.load(path.open('r'))
        hparams['created_by_merging'] = [str(indir) for indir in args.indirs]
        json.dump(hparams, new_path.open('w'), indent=2)
        print(path, '-->', new_path)

    for mode in ['train', 'test', 'val']:
        files = []
        for in_dir in args.indirs:
            mode_indir = in_dir / mode
            tfrecord_files = mode_indir.glob("*.tfrecords")
            files.extend(tfrecord_files)

        traj_idx = 0
        for i, file in enumerate(files):
            path = pathlib.Path(file)
            filename = path.name
            m = re.match(r".*?_(\d+)_to_(\d+).tfrecords", filename)
            start, end = m.group(1), m.group(2)
            n_trajs_in_file = int(end) - int(start)
            new_filename = "traj_{}_to_{}.tfrecords".format(traj_idx, traj_idx + n_trajs_in_file)
            mode_outdir = args.outdir / mode
            mode_outdir.mkdir(parents=True, exist_ok=True)
            new_path = pathlib.Path(mode_outdir) / new_filename
            traj_idx = traj_idx + n_trajs_in_file + 1
            print(path, '-->', new_path)
            if not args.dry_run:
                shutil.copyfile(path, new_path)


if __name__ == '__main__':
    main()
