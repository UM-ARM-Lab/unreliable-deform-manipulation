import pathlib
import argparse


def main():
    parser = argparse.ArgumentParser("convert a dynamics dataset into a classifier dataset by running a model on each sequence")
    parser.add_argument("dataset_dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("fwd_model_dir", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)

    args = parser.parse_args()

    n_examples_per_record = 128
    compression_type = "ZLIB"
    current_record_idx = 0

    fwd_model, _ = model_utils.load_generic_fwd_model(args.fwd_model_dir)

    dataset = LinkBotStateSpaceDataset(args.dataset_dirs)

    states_description = dataset.states_description
    actual_state_keys = states_description.keys()
    planned_state_keys = fwd_model.hparams['states_keys']

    new_hparams = dataset.hparams
    new_hparams.pop('states_description'])
    new_hparams['actual_state_keys'] = actual_state_keys 
    new_hparams['planned_state_keys'] = planned_state_keys


    for mode in ['test', 'val', 'train']:
        full_output_directory = args.out_dir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        tf_dataset = dataset.get_dataset(mode=mode)

        examples = np.ndarray([n_examples_per_record], dtype=np.object)
        for example_idx, example_dict in enumerate(tf_dataset):

            actions = input_data['action'].numpy()

            full_env = input_data['full_env/env'].numpy()
            full_env_origin = input_data['full_env/origin'].numpy()
            full_env_extents = input_data['full_env/extent'].numpy()
            full_env_res = input_data['full_env/res'].numpy()

            for state_key, n_state in states_description.items():
                states = input_data[state_key].numpy()

            predictions = fwd_model.predict(full_env=,
                    full_env_origin=,
                    start_states=,
                    actions=,)

            features = {
                'full_env': float_tensor_to_bytes_feature(full_env),
                'full_env/origin': float_tensor_to_bytes_feature(full_env_origin),
                'full_env/extent': float_tensor_to_bytes_feature(full_env_extent),
                'full_env/res': float_tensor_to_bytes_feature(full_env_res),
            }

            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            example = example_proto.SerializeToString()
            examples[current_record_idx] = example
            current_record_idx += 1

            if current_record_idx == n_examples_per_record:
                # save to a TF record
                serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

                end_example_idx = example_idx + 1
                start_example_idx = end_example_idx - n_examples_per_record
                record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
                full_filename = full_output_directory / record_filename
                writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=compression_type)
                writer.write(serialized_dataset)
                print("saved {}".format(full_filename))

                current_record_idx = 0
        return

    new_hparams_path = outargs.outdir / 'hparams.json'
    with new_hparams_path.open("w") as oh:
        json.dump(new_haparams, oh)


if __name__ == '__main__':
    main()
