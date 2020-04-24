#!/usr/bin/env python
import argparse
import numpy as np
import json
import pathlib

from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_sdf_utils import compute_extent
from moonshine.numpy_utils import listify


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("metrics", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)

    args = parser.parse_args()

    new_metric_file = args.outdir / 'metrics.json'
    if new_metric_file.exists():
        print("new metric file already exists -- aborting!")
        return
    args.outdir.mkdir(exist_ok=True, parents=True)

    old_metrics_filename = args.metrics / 'metrics.json'
    old = json.load(old_metrics_filename.open('r'))
    old_metrics = old.pop("metrics")
    new = old

    new_metrics = []
    for old_metric in old_metrics:
        env = old_metric.pop("full_env")
        h, w = np.array(env).shape
        origin = np.array([h/2, w/2], dtype=np.int)
        res = 0.01
        extent = compute_extent(h, w, res, origin)
        new_metric = old_metric
        new_metric['environment'] = listify({
            'full_env/env': env,
            'full_env/extent': extent,
            'full_env/res': res,
            'full_env/origin': origin,
        })
        new_metrics.append(new_metric)

    new['metrics'] = new_metrics

    print("Saving to {}".format(new_metric_file))
    json.dump(new, new_metric_file.open("w"), indent=2)


if __name__ == '__main__':
    main()
