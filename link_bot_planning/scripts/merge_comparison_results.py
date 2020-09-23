#!/usr/bin/env python
import argparse
import pathlib
import shutil

import colorama
from colorama import Fore

from link_bot_pycommon.args import my_formatter


def main():
    colorama.init(autoreset=True)

    # Copies without overwriting by incrementing index, and merges json file
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('subdir', type=pathlib.Path, help="subdir")
    parser.add_argument('out_subdir', type=pathlib.Path, help="combined data will go here")
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    old_metrics_filenames = list(args.subdir.glob("*_metrics.json.gz"))
    existing_metrics_filenames = list(args.out_subdir.glob("*_metrics.json.gz"))
    # start off where existing metrics left off
    new_idx = len(existing_metrics_filenames)
    for old_metrics_filename in old_metrics_filenames:
        new_metrics_filename = args.out_subdir / f"{new_idx}_metrics.json.gz"
        if new_metrics_filename.exists():
            print(Fore.RED + f"Refusing to overwrite {new_metrics_filename}. Aborting." + Fore.RESET)
            return
        else:
            print(f"Copying {old_metrics_filename} -> {new_metrics_filename}")
            if not args.dry_run:
                shutil.copy(old_metrics_filename, new_metrics_filename)
        new_idx += 1


if __name__ == '__main__':
    main()
