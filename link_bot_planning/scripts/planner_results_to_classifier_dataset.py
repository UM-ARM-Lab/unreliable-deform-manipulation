#!/usr/bin/env python
import argparse
import json
import pathlib
from time import perf_counter

import tensorflow as tf
from colorama import Fore

from link_bot_data.classifier_dataset_utils import generate_classifier_examples_from_batch
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature
from link_bot_pycommon.args import my_formatter
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_np_arrays

limit_gpu_mem(1)


def plan_result_to_examples(result_idx, result, labeling_params):
    actual_path = result['actual_path']
    actions = result['actions']
    planned_path = result['planned_path']
    environment = result['environment']
    classifier_horizon = labeling_params['classifier_horizon']
    for prediction_start_t in range(0, len(actual_path) - classifier_horizon - 1, labeling_params['start_step']):
        inputs = {
            'traj_idx': tf.cast([result_idx] * classifier_horizon, tf.float32),
            'action': actions[prediction_start_t:],
        }
        inputs.update(environment)
        outputs = sequence_of_dicts_to_dict_of_np_arrays(actual_path[prediction_start_t:])
        predictions = sequence_of_dicts_to_dict_of_np_arrays(planned_path[prediction_start_t:])
        yield from generate_classifier_examples_from_batch()


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('outdir', type=pathlib.Path)
    parser.add_argument('--max-examples-per-record', type=int, default=512)

    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True)

    compression_type = 'ZLIB'
    labeling_params = json.load(args.labeling_params.open("r"))

    for results_dir in args.results_dirs:
        results_filename = results_dir / 'metrics.json'
        results = json.load(results_filename.open("r"))
        planner_params = results['planner_params']
        fwd_model_dir = planner_params['fwd_model_dir'][0]
        fwd_model_hparams_filename = pathlib.Path(fwd_model_dir) / 'hparams.json'
        fwd_model_hparams = json.load(fwd_model_hparams_filename.open('r'))
        # copy hparams from dynamics dataset into classifier dataset

        new_hparams_filename = args.outdir / 'hparams.json'
        classifier_dataset_hparams = {}
        classifier_dataset_hparams['dataset_dir'] = [d.as_posix() for d in args.results_dirs]
        classifier_dataset_hparams['fwd_model_dir'] = fwd_model_dir
        for k, v in fwd_model_hparams['dynamics_dataset_hparams'].items():
            classifier_dataset_hparams[k] = v
        classifier_dataset_hparams['fwd_model_hparams'] = fwd_model_hparams
        classifier_dataset_hparams['labeling_params'] = labeling_params
        classifier_dataset_hparams['state_keys'] = [fwd_model_hparams['state_key']]
        json.dump(classifier_dataset_hparams, new_hparams_filename.open("w"), indent=2)

        last_record = perf_counter()
        current_example_count = 0
        examples = []
        total_count = 0
        for result_idx, result in enumerate(results['metrics']):
            for example_idx, out_example in enumerate(plan_result_to_examples(result_idx, result, labeling_params)):
                features = {}
                for k, v in out_example.items():
                    features[k] = float_tensor_to_bytes_feature(v)

                example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                example = example_proto.SerializeToString()
                examples.append(example)
                current_example_count += 1
                total_count += 1

                if current_example_count == args.max_examples_per_record:
                    # save to a TF record
                    serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

                    end_example_idx = total_count
                    start_example_idx = end_example_idx - len(examples)
                    record_filename = "example_{:09d}_to_{:09d}.tfrecords".format(start_example_idx, end_example_idx - 1)
                    full_filename = args.outdir / record_filename
                    if full_filename.exists():
                        print(Fore.RED + "Error! Output file {} exists. Aborting.".format(full_filename) + Fore.RESET)
                        return
                    writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=compression_type)
                    writer.write(serialized_dataset)
                    now = perf_counter()
                    dt_record = now - last_record
                    print("saved {} ({:.3f}s)".format(full_filename, dt_record))
                    last_record = now

                    # empty and reset counter
                    current_example_count = 0
                    examples = []


if __name__ == '__main__':
    main()
