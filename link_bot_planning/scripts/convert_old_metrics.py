#!/usr/bin/env python
import argparse
import json
import pathlib

from link_bot_pycommon.args import my_formatter


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)

    parser.add_argument('results_dir', type=pathlib.Path)
    parser.add_argument('new_results_dir', type=pathlib.Path)

    args = parser.parse_args()
    args.new_results_dir.mkdir()

    for subdir in args.results_dir.iterdir():
        old_json_filename = subdir / 'metrics.json'
        old_json = json.load(old_json_filename.open("r"))
        new_json = old_json
        for metric in new_json['metrics']:
            metric['environment'] = {
                'full_env/env': metric['full_env'],
                'full_env/origin': [100.0, 100.0],
                'full_env/extent': [-1.0, 1.0, -1.0, 1.0],
                'full_env/res': 0.01,
            }

        new_subdir = args.new_results_dir / subdir.name
        new_subdir.mkdir()
        new_json_filename = new_subdir / 'metrics.json'
        print(new_json_filename)
        json.dump(new_json, new_json_filename.open("w"))


if __name__ == '__main__':
    main()
