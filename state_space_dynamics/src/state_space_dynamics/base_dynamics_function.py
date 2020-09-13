import pathlib
from typing import Dict, List, Tuple

import tensorflow as tf
from colorama import Fore

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import make_dict_tf_float32
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors, add_batch, remove_batch, \
    dict_of_sequences_to_sequence_of_dicts_tf, numpify
from shape_completion_training.model.filepath_tools import load_trial


class BaseDynamicsFunction:

    @staticmethod
    def get_net_class():
        raise NotImplementedError()

    def __init__(self, model_dirs: List[pathlib.Path], batch_size: int, scenario: ExperimentScenario):
        representative_model_dir = model_dirs[0]
        _, self.hparams = load_trial(representative_model_dir.parent.absolute())

        self.scenario = scenario
        self.batch_size = batch_size
        self.data_collection_params = self.hparams['dynamics_dataset_hparams']['data_collection_params']
        self.max_step_size = self.data_collection_params['max_step_size']
        self.states_description = self.hparams['dynamics_dataset_hparams']['states_description']
        self.action_description = self.hparams['dynamics_dataset_hparams']['action_description']

        net_class_name = self.get_net_class()

        self.nets = []
        for model_dir in model_dirs:
            net = net_class_name(hparams=self.hparams, batch_size=batch_size, scenario=scenario)
            ckpt = tf.train.Checkpoint(model=net)
            manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=1)

            status = ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
                if manager.latest_checkpoint:
                    status.assert_existing_objects_matched()
            else:
                raise RuntimeError("Failed to restore!!!")

            self.nets.append(net)

            self.state_keys = net.state_keys
            self.action_keys = net.action_keys

    def propagate_from_example(self, example, training: bool = False):
        """ This is the function that all other propagate functions eventually call """
        if 'batch_size' not in example:
            example['batch_size'] = example[self.state_keys[0]].shape[0]
        if 'sequence_length' not in example:
            example['sequence_length'] = example[self.action_keys[0]].shape[1] + 1  # only used by Image Cond Dyn
        predictions = [net(example, training=training) for net in self.nets]
        predictions_dict = sequence_of_dicts_to_dict_of_tensors(predictions)
        mean_prediction = {state_key: tf.math.reduce_mean(predictions_dict[state_key], axis=0) for state_key in
                           self.state_keys}
        stdev_prediction = {state_key: tf.math.reduce_std(predictions_dict[state_key], axis=0) for state_key in
                            self.state_keys}
        all_stdevs = tf.concat(list(stdev_prediction.values()), axis=2)
        mean_prediction['stdev'] = tf.reduce_sum(all_stdevs, axis=2, keepdims=True)
        return mean_prediction, stdev_prediction

    def propagate(self, environment: Dict, start_states: Dict, actions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        mean_predictions, stdev_predictions = self.propagate_differentiable(environment, start_states, actions)
        return numpify(mean_predictions), numpify(stdev_predictions)

    def propagate_differentiable(self, environment: Dict, start_states: Dict, actions: List[Dict]) -> Tuple[
        List[Dict], List[Dict]]:
        # add time dimension of size 1
        net_inputs = {k: tf.expand_dims(start_states[k], axis=0) for k in self.state_keys}
        net_inputs.update(sequence_of_dicts_to_dict_of_tensors(actions))
        net_inputs.update(environment)
        net_inputs = add_batch(net_inputs)
        net_inputs = make_dict_tf_float32(net_inputs)
        # the network returns a dictionary where each value is [T, n_state]
        # which is what you'd want for training, but for planning and execution and everything else
        # it is easier to deal with a list of states where each state is a dictionary
        mean_predictions, stdev_predictions = self.propagate_from_example(net_inputs, training=False)
        mean_predictions = remove_batch(mean_predictions)
        stdev_predictions = remove_batch(stdev_predictions)
        mean_predictions = dict_of_sequences_to_sequence_of_dicts_tf(mean_predictions)
        stdev_predictions = dict_of_sequences_to_sequence_of_dicts_tf(stdev_predictions)
        return mean_predictions, stdev_predictions

    def propagate_differentiable_batched(self, environment: Dict, states: Dict, actions: Dict) -> Tuple[Dict, Dict]:
        net_inputs = states
        net_inputs.update(actions)
        net_inputs.update(environment)
        net_inputs = make_dict_tf_float32(net_inputs)
        mean_predictions, stdev_predictions = self.propagate_from_example(net_inputs, training=False)
        return mean_predictions, stdev_predictions
