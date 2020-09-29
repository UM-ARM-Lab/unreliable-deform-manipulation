#!/usr/bin/env python
import argparse
import gzip
import json
import pathlib
import shutil
from copy import deepcopy

import colorama
from colorama import Fore

from link_bot_pycommon.args import my_formatter


def main():
    # Use to update the format of results when I make breaking changes to the code
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dir', help='results directory', type=pathlib.Path)

    args = parser.parse_args()

    for in_results_subdir in args.results_dir.iterdir():
        if not in_results_subdir.is_dir():
            continue
        print(Fore.BLUE + in_results_subdir.as_posix())
        metrics_filenames = list(in_results_subdir.glob("*_metrics.json.gz"))
        out_results_dir = args.results_dir.parent / (args.results_dir.name + '_updated')
        in_metdata_file = in_results_subdir / 'metadata.json'
        out_subdir = out_results_dir / in_results_subdir.name
        out_metadata_file = out_subdir / 'metadata.json'
        shutil.copy(in_metdata_file, out_metadata_file)
        out_subdir.mkdir(parents=True, exist_ok=True)

        for plan_idx, metrics_filename in enumerate(metrics_filenames):
            with gzip.open(metrics_filename, 'rb') as metrics_file:
                data_str = metrics_file.read()
            datum = json.loads(data_str.decode("utf-8"))

            datum['end_state']['rope'] = datum['end_state']['link_bot']
            out_filename = out_subdir / f'{plan_idx}_metrics.json.gz'
            out_data_str = json.dumps(datum)
            print(f"writing {out_filename}")
            with gzip.open(out_filename, 'wb') as out_file:
                out_file.write(out_data_str.encode("utf-8"))


if __name__ == '__main__':
    main()
