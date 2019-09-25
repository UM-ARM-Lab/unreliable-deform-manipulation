#!/usr/bin/env python
import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=pathlib.Path)

    args = parser.parse_args()


if __name__ == '__main__':
    main()
