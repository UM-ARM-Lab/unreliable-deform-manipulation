#!/usr/bin/env python
import pathlib

import shutil
import gzip
import json
import argparse

from colorama import Fore

from link_bot_pycommon.args import my_formatter


def main():
    # Copies without overwriting by incrementing index, and merges json file
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('subdir', type=pathlib.Path, help="subdir")
    parser.add_argument('out_subdir', type=pathlib.Path, help="combined data will go here")
    args = parser.parse_args()

    in_json = args.subdir / 'metrics.json.gz'
    with gzip.open(in_json, 'rb') as in_json_f:
        in_json_str = in_json_f.read()
        in_metrics = json.loads(in_json_str.decode("utf-8"))

    args.out_subdir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_subdir / 'metrics.json.gz'

    if out_json.exists():
        # FIXME: continue converting this script
        with out_json.open("r") as out_json_f_read:
            out_metrics = json.load(out_json_f_read)
            # assumes the parameters are the same, only copies the metrics, but will warn on differences
            new_out_metrics = {}
            for (k, v1), (_, v2) in zip(out_metrics.items(), in_metrics.items()):
                if k != 'metrics':
                    new_out_metrics[k] = v1
                    if k != 'n_total_plans' and k != 'initial_poses_in_collision':
                        # warn on differences
                        if v1 != v2:
                            warning = "WARNING: hyperparmeter value of {} differs:\n{}\n{}\n using the former".format(
                                k, v1, v2)
                            print(Fore.YELLOW + warning + Fore.RESET)

            in_metrics_list = in_metrics['metrics']
            out_metrics_list = out_metrics['metrics']

            # ensure no duplicates!
            new_metrics_list = out_metrics_list
            for in_idx, in_metric in enumerate(in_metrics_list):
                for o in out_metrics_list:
                    if in_metric == o:
                        print("ERROR! there are duplicates. ABORTING")
                        return
                new_metrics_list.append(in_metric)
                # copy the plan image files, renaming if necessary
                out_idx = len(new_metrics_list) - 1
                in_img = args.subdir / 'plan_{:d}.png'.format(in_idx)
                if in_img.exists():
                    out_img = args.out_subdir / 'plan_{:d}.png'.format(out_idx)
                    shutil.copy2(in_img, out_img)
                else:
                    print(Fore.YELLOW + "Missing plan image {}".format(in_img))
            new_out_metrics['n_total_plans'] = len(new_metrics_list)
            new_out_metrics['metrics'] = new_metrics_list
            tmp_out_json = pathlib.Path('/tmp/tmp_metrics.json')
        with tmp_out_json.open("w") as tmp_out_json_f_write:
            json.dump(new_out_metrics, tmp_out_json_f_write, indent=2)
        shutil.copy2(tmp_out_json, out_json)
    else:
        with out_json.open("w+") as out_json_f:
            out_metrics = in_metrics
            json.dump(out_metrics, out_json_f, indent=2)

        for in_img in args.subdir.glob("plan_*.png"):
            if in_img.exists():
                out_img = args.out_subdir / in_img.name
                shutil.copy2(in_img, out_img)
            else:
                print(Fore.YELLOW + "Missing plan image {}".format(in_img))


if __name__ == '__main__':
    main()
