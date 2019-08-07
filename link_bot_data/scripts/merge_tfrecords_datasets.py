import argparse
import glob
from colorama import Fore
import os
import shutils

def main():
    parser = argparse.ArgumentParser()
    parser.add_arugment("indirs", nargs="?")
    parser.add_argument("outdir")

    args = parser.parse_args()

    for indir in args.indirs:
        if not os.path.isdir(indir):
            print(Fore.YELLOW + "{} is not a directory".format(indir) + Fore.RESET)
        tfrecord_files = glob.glob(indir + "/*.tfrecords")
        print(tfrecord_files)


if __name__ == '__main__':
    main()

