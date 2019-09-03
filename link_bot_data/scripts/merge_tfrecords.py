#!/usr/bin/env python

import argparse
import glob
import os
import pathlib
import re
import shutil

from colorama import Fore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indirs", nargs="*")
    parser.add_argument("--outdir")
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()

    if args.outdir and not os.path.isdir(args.outdir):
        print(Fore.YELLOW + "{} is not a directory".format(args.outdir) + Fore.RESET)
        return

    files = []
    for in_dir in args.indirs:
        if not os.path.isdir(in_dir):
            print(Fore.YELLOW + "{} is not a directory".format(in_dir) + Fore.RESET)
            return
        tfrecord_files = glob.glob(in_dir + "/*.tfrecords")
        files.extend(tfrecord_files)

    traj_idx = 0
    for i, file in enumerate(files):
        path = pathlib.Path(file)
        parent = path.parent
        filename = path.name
        m = re.match(r"traj_(\d+)_to_(\d+).tfrecords", filename)
        start, end = m.group(1), m.group(2)
        n_trajs_in_file = int(end) - int(start)
        new_filename = "traj_{}_to_{}.tfrecords".format(traj_idx, traj_idx + n_trajs_in_file)
        new_path = pathlib.Path(args.outdir) / new_filename
        traj_idx = traj_idx + n_trajs_in_file + 1
        print(path, '-->',  new_path)
        if not args.dry_run:
            shutil.copyfile(path, new_path)


if __name__ == '__main__':
    main()
