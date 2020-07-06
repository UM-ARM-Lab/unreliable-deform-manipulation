#!/usr/bin/env python

from copy import deepcopy
import json
import pathlib
import argparse
import gzip

from link_bot_pycommon.args import my_formatter


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_subdir', help='results directory', type=pathlib.Path)

    args = parser.parse_args()

    metrics_filenames = list(args.results_subdir.glob("99_metrics.json.gz"))
    for plan_idx, metrics_filename in enumerate(metrics_filenames):
        print(plan_idx)
        with gzip.open(metrics_filename, 'rb') as metrics_file:
            data_str = metrics_file.read()
        datum = json.loads(data_str.decode("utf-8"))

        out_data = deepcopy(datum)
        out_data.update(datum['metrics'])
        out_filename = args.results_subdir / f'99_metrics.json.gz'
        out_data_str = json.dumps(out_data)
        with gzip.open(out_filename, 'wb') as out_file:
            out_file.write(out_data_str.encode("utf-8"))


if __name__ == '__main__':
    main()
