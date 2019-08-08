#!/usr/bin/env python
import glob
import argparse
import os

import tensorflow as tf
tf.enable_eager_execution()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir')
    parser.add_argument('outdir')
    parser.add_argument('in_compression_type', choices=['', 'ZLIB', 'GZIP'])
    parser.add_argument('out_compression_type', choices=['', 'ZLIB', 'GZIP'])

    args = parser.parse_args()
    recompress(args.indir, args.outdir, args.in_compression_type, args.out_compression_type)


def recompress(indir, outdir, in_compression_type, out_compression_type):
    files = glob.glob(os.path.join(indir, "*.tfrecords"))
    for file in files:
        filename = os.path.basename(file)
        dataset = tf.data.TFRecordDataset(file, compression_type=in_compression_type)

        out_file = os.path.join(outdir, filename)
        writer = tf.data.experimental.TFRecordWriter(out_file, compression_type=out_compression_type)
        writer.write(dataset)
        print("Wrote ", out_file)


if __name__ == '__main__':
    main()
