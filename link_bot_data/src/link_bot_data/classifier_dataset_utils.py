import json
import pathlib
from time import perf_counter
from typing import Dict, List, Optional

import hjson
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import add_predicted, batch_tf_dataset, float_tensor_to_bytes_feature, \
    add_label
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import index_dict_of_batched_vectors_tf
from state_space_dynamics import model_utils
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class PredictionActualExample:
    def __init__(self,
                 example: Dict,
                 actual_states: Dict,
                 actions: Dict,
                 predictions: Dict,
                 start_t: int,
                 labeling_params: Dict,
                 actual_prediction_horizon: int,
                 batch_size: int):
        self.dataset_element = example
        self.actual_states = actual_states
        self.actions = actions
        self.predictions = predictions
        self.prediction_start_t = start_t
        self.labeling_params = labeling_params
        self.actual_prediction_horizon = actual_prediction_horizon
        self.batch_size = batch_size


def make_classifier_dataset(dataset_dir: pathlib.Path,
                            fwd_model_dir: List[pathlib.Path],
                            labeling_params: pathlib.Path,
                            outdir: pathlib.Path,
                            use_gt_rope: bool,
                            start_at: Optional[int] = None,
                            stop_at: Optional[int] = None):
    labeling_params = json.load(labeling_params.open("r"))
    make_classifier_dataset_from_params_dict(dataset_dir, fwd_model_dir, labeling_params, outdir, use_gt_rope, start_at,
                                             stop_at)


def make_classifier_dataset_from_params_dict(dataset_dir: pathlib.Path,
                                             fwd_model_dir: List[pathlib.Path],
                                             labeling_params: Dict,
                                             outdir: pathlib.Path,
                                             use_gt_rope: bool,
                                             start_at: Optional[int] = None,
                                             stop_at: Optional[int] = None):
    # append "best_checkpoint" before loading
    if not isinstance(fwd_model_dir, List):
        fwd_model_dir = [fwd_model_dir]
    fwd_model_dir = [p / 'best_checkpoint' for p in fwd_model_dir]

    dynamics_hparams = hjson.load((dataset_dir / 'hparams.hjson').open('r'))
    fwd_models, _ = model_utils.load_generic_model(fwd_model_dir)

    dataset = DynamicsDataset([dataset_dir], use_gt_rope=use_gt_rope)

    new_hparams_filename = outdir / 'hparams.hjson'
    classifier_dataset_hparams = dynamics_hparams

    classifier_dataset_hparams['dataset_dir'] = dataset_dir.as_posix()
    classifier_dataset_hparams['fwd_model_hparams'] = fwd_models.hparams
    classifier_dataset_hparams['labeling_params'] = labeling_params
    classifier_dataset_hparams['true_state_keys'] = dataset.state_keys
    classifier_dataset_hparams['predicted_state_keys'] = fwd_models.state_keys
    classifier_dataset_hparams['action_keys'] = dataset.action_keys
    classifier_dataset_hparams['scenario_metadata'] = dataset.hparams['scenario_metadata']
    classifier_dataset_hparams['start-at'] = start_at
    classifier_dataset_hparams['stop-at'] = stop_at
    hjson.dump(classifier_dataset_hparams, new_hparams_filename.open("w"), indent=2)

    # because we're currently making this dataset, we can't call "get_dataset" but we can still use it to visualize
    classifier_dataset_for_viz = ClassifierDataset([outdir], use_gt_rope=use_gt_rope)

    t0 = perf_counter()
    total_example_idx = 0
    for mode in ['train', 'val', 'test']:
        tf_dataset = dataset.get_datasets(mode=mode)

        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        batch_size = 8
        out_examples = generate_classifier_examples(fwd_models, tf_dataset, dataset, labeling_params, batch_size)
        for out_example in out_examples:
            actual_batch_size = out_example[0]['traj_idx'].shape[0]
            for batch_idx in range(actual_batch_size):
                for out_example_for_start_t in out_example:
                    out_example_b = index_dict_of_batched_vectors_tf(out_example_for_start_t, batch_idx)

                    DEBUG = True
                    if DEBUG:
                        add_label(out_example_b, labeling_params['threshold'])
                        classifier_dataset_for_viz.plot_transition_rviz(out_example_b)

                    features = {k: float_tensor_to_bytes_feature(v) for k, v in out_example_b.items()}

                    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                    example = example_proto.SerializeToString()
                    record_filename = "example_{:09d}.tfrecords".format(total_example_idx)
                    full_filename = full_output_directory / record_filename
                    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')
                    with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
                        writer.write(example)
                    print(f"Examples: {total_example_idx:10d}, Time: {perf_counter() - t0:.3f}")
                    total_example_idx += 1

    return outdir


def generate_classifier_examples(fwd_model: BaseDynamicsFunction,
                                 tf_dataset: tf.data.Dataset,
                                 dataset: DynamicsDataset,
                                 labeling_params: Dict,
                                 batch_size: int):
    classifier_horizon = labeling_params['classifier_horizon']
    assert classifier_horizon >= 2
    tf_dataset = batch_tf_dataset(tf_dataset, batch_size, drop_remainder=False)
    sc = dataset.scenario

    for idx, _ in enumerate(tf_dataset):
        pass
    n_total_batches = idx

    t0 = perf_counter()
    for idx, example in enumerate(tf_dataset):
        dt = perf_counter() - t0
        print(f"{idx} / {n_total_batches} batches in {dt:.3f} seconds")
        actual_batch_size = int(example['traj_idx'].shape[0])

        valid_out_examples = []
        for start_t in range(0, dataset.steps_per_traj - classifier_horizon + 1, labeling_params['start_step']):
            prediction_end_t = dataset.steps_per_traj
            actual_prediction_horizon = prediction_end_t - start_t
            actual_states_from_start_t = {k: example[k][:, start_t:prediction_end_t] for k in dataset.state_keys}
            actions_from_start_t = {k: example[k][:, start_t:prediction_end_t - 1] for k in dataset.action_keys}

            predictions_from_start_t, _ = fwd_model.propagate_differentiable_batched(environment={},
                                                                                     state=actual_states_from_start_t,
                                                                                     actions=actions_from_start_t)
            prediction_actual = PredictionActualExample(example=example,
                                                        actions=actions_from_start_t,
                                                        actual_states=actual_states_from_start_t,
                                                        predictions=predictions_from_start_t,
                                                        start_t=start_t,
                                                        labeling_params=labeling_params,
                                                        actual_prediction_horizon=actual_prediction_horizon,
                                                        batch_size=actual_batch_size)
            valid_out_examples_for_start_t = generate_classifier_examples_from_batch(sc, prediction_actual)
            valid_out_examples.extend(valid_out_examples_for_start_t)

        yield valid_out_examples


def generate_classifier_examples_from_batch(scenario: ExperimentScenario, prediction_actual: PredictionActualExample):
    labeling_params = prediction_actual.labeling_params
    prediction_horizon = prediction_actual.actual_prediction_horizon
    classifier_horizon = labeling_params['classifier_horizon']

    valid_out_examples = []
    for classifier_start_t in range(0, prediction_horizon - classifier_horizon + 1):
        classifier_end_t = classifier_start_t + classifier_horizon

        prediction_start_t = prediction_actual.prediction_start_t
        prediction_start_t_batched = tf.cast(
            tf.stack([prediction_start_t] * prediction_actual.batch_size, axis=0), tf.float32)
        classifier_start_t_batched = tf.cast(
            tf.stack([classifier_start_t] * prediction_actual.batch_size, axis=0), tf.float32)
        classifier_end_t_batched = tf.cast(
            tf.stack([classifier_end_t] * prediction_actual.batch_size, axis=0), tf.float32)
        out_example = {
            'env': prediction_actual.dataset_element['env'],
            'origin': prediction_actual.dataset_element['origin'],
            'extent': prediction_actual.dataset_element['extent'],
            'res': prediction_actual.dataset_element['res'],
            'traj_idx': prediction_actual.dataset_element['traj_idx'],
            'prediction_start_t': prediction_start_t_batched,
            'classifier_start_t': classifier_start_t_batched,
            'classifier_end_t': classifier_end_t_batched,
        }

        # this slice gives arrays of fixed length (ex, 5) which must be null padded from out_example_end_idx onwards
        state_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon)
        action_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon - 1)
        sliced_actual = {}
        for key, actual_state_component in prediction_actual.actual_states.items():
            actual_state_component_sliced = actual_state_component[:, state_slice]
            out_example[key] = actual_state_component_sliced
            sliced_actual[key] = actual_state_component_sliced

        sliced_predictions = {}
        for key, prediction_component in prediction_actual.predictions.items():
            prediction_component_sliced = prediction_component[:, state_slice]
            out_example[add_predicted(key)] = prediction_component_sliced
            sliced_predictions[key] = prediction_component_sliced

        # action
        sliced_actions = {}
        for key, action_component in prediction_actual.actions.items():
            action_component_sliced = action_component[:, action_slice]
            out_example[key] = action_component_sliced
            sliced_actions[key] = action_component_sliced

        # compute label
        threshold = labeling_params['threshold']
        error = scenario.classifier_distance(sliced_actual, sliced_predictions)
        out_example['error'] = tf.cast(error, dtype=tf.float32)

        # is_first_predicted_state_close = is_close[:, 0]
        # valid_indices = tf.where(is_first_predicted_state_close)
        # valid_indices = tf.squeeze(valid_indices, axis=1)
        # # keep only valid_indices from every key in out_example...
        # valid_out_example = gather_dict(out_example, valid_indices)
        # valid_out_examples.append(valid_out_example)

        valid_out_examples.append(out_example)
    return valid_out_examples


def batch_of_many_of_actions_sequences_to_dict(actions, n_actions_sampled, n_start_states, n_actions):
    # reformat the inputs to be efficiently batched
    actions_dict = {}
    for actions_for_start_state in actions:
        for actions in actions_for_start_state:
            for action in actions:
                for k, v in action.items():
                    if k not in actions_dict:
                        actions_dict[k] = []
                    actions_dict[k].append(v)
    actions_batched = {k: tf.reshape(v, [n_actions_sampled * n_start_states, n_actions, -1])
                       for k, v in actions_dict.items()}
    return actions_batched
