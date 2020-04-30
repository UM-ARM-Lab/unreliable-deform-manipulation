#!/usr/bin/env python

import argparse
import json
import pathlib

import matplotlib.pyplot as plt

from link_bot_planning.get_scenario import get_scenario
from link_bot_pycommon.args import my_formatter


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('logfile', type=pathlib.Path)

    args = parser.parse_args()

    logged_data = json.load(args.logfile.open("r"))

    scenario = get_scenario(logged_data['scenario'])
    anim = scenario.animate_predictions(environment=logged_data['environment'],
                                        actual=logged_data['path'],
                                        actions=logged_data['actions'],
                                        predictions=None)
    anim.save(filename='out.gif', writer='imagemagick', dpi=300, fps=10)
    plt.show()


if __name__ == '__main__':
    main()
