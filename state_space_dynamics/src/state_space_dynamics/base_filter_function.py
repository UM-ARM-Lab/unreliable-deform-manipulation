import pathlib
from typing import Dict, List, Tuple, Optional

import tensorflow as tf
from colorama import Fore

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import make_dict_tf_float32
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors, add_batch, remove_batch, \
    dict_of_sequences_to_sequence_of_dicts_tf, numpify
from shape_completion_training.model.filepath_tools import load_trial
from shape_completion_training.my_keras_model import MyKerasModel


class BaseFilterFunction:

    @staticmethod
    def get_net_class():
        raise NotImplementedError()

    def __init__(self, model_dirs: List[pathlib.Path], batch_size: int, scenario: ExperimentScenario):
        representative_model_dir = model_dirs[0]
        _, self.hparams = load_trial(representative_model_dir.parent.absolute())

        self.scenario = scenario
        self.batch_size = batch_size
        self.data_collection_params = self.hparams['dynamics_dataset_hparams']['data_collection_params']

        net_class_name = self.get_net_class()

        self.nets: List[MyKerasModel] = []
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
            self.obs_keys = net.obs_keys
            self.action_keys = net.action_keys

    def filter_from_example(self, example, training: bool = False):
        """ This is the function that all other filter functions eventually call """
        if 'batch_size' not in example:
            example['batch_size'] = example[self.obs_keys[0]].shape[0]
        filtered_states = [net(net.preprocess_no_gradient(example), training=training) for net in self.nets]
        filtered_states_dict = sequence_of_dicts_to_dict_of_tensors(filtered_states)
        mean_state = {state_key: tf.math.reduce_mean(filtered_states_dict[state_key], axis=0) for state_key in self.state_keys}
        stdev_state = {state_key: tf.math.reduce_std(filtered_states_dict[state_key], axis=0) for state_key in self.state_keys}
        all_stdevs = tf.concat(list(stdev_state.values()), axis=2)
        mean_state['stdev'] = tf.reduce_sum(all_stdevs, axis=2, keepdims=True)
        return mean_state, stdev_state

    def filter(self, environment: Dict, state: Optional[Dict], observation: Dict) -> Tuple[Dict, Dict]:
        mean_state, stdev_state = self.filter_differentiable(environment, state, observation)
        return numpify(mean_state), numpify(stdev_state)

    def filter_differentiable(self, environment: Dict, state: Optional[Dict], observation: Dict) -> Tuple[Dict, Dict]:
        # add time dimension of size 1
        net_inputs = {k: tf.expand_dims(state[k], axis=0) for k in self.state_keys}
        net_inputs.update({k: tf.expand_dims(observation[k], axis=0) for k in self.obs_keys})
        net_inputs.update(environment)
        net_inputs = add_batch(net_inputs)
        net_inputs = make_dict_tf_float32(net_inputs)

        mean_state, stdev_state = self.filter_from_example(net_inputs, training=False)
        mean_state = remove_batch(mean_state)
        stdev_state = remove_batch(stdev_state)
        mean_state = dict_of_sequences_to_sequence_of_dicts_tf(mean_state)
        stdev_state = dict_of_sequences_to_sequence_of_dicts_tf(stdev_state)
        return mean_state, stdev_state

    def filter_differentiable_batched(self, environment: Dict, state: Optional[Dict], observation: Dict) -> Tuple[Dict, Dict]:
        net_inputs = state
        net_inputs.update(observation)
        net_inputs.update(environment)
        net_inputs = make_dict_tf_float32(net_inputs)
        mean_predictions, stdev_predictions = self.filter_from_example(net_inputs, training=False)
        return mean_predictions, stdev_predictions
