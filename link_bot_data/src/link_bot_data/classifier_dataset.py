import pathlib
from typing import List, Dict

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import add_next, convert_sequences_to_transitions, add_planned
from link_bot_planning.params import FullEnvParams
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def add_model_predictions(fwd_model: BaseDynamicsFunction, tf_dataset, dataset: DynamicsDataset):
    def _add_model_predictions(inputs, outputs):
        full_env = inputs['full_env/env']
        full_env_origin = inputs['full_env/origin']
        res = inputs['full_env/res']
        actions = inputs['action']
        start_states = {}
        for name in dataset.states_description.keys():
            start_state = inputs[name][0]
            start_states[name] = start_state

        predictions = fwd_model.propagate_differentiable(full_env=full_env,
                                                         full_env_origin=full_env_origin,
                                                         res=res,
                                                         start_states=start_states,
                                                         actions=actions)
        for name in fwd_model.states_keys:
            predictions_for_name = []
            for prediction_t in predictions:
                predictions_for_name.append(prediction_t[name])
            predictions_for_name = tf.stack(predictions_for_name, axis=0)
            outputs[add_planned(name)] = predictions_for_name
        return inputs, outputs

    def _split_sequences(inputs, outputs):
        example = {}
        for name in dataset.action_feature_names:
            example[name] = inputs[name]

        for name in dataset.state_feature_names:
            # use outputs here so we get the final state
            example[name] = outputs[name]

        for name in fwd_model.states_keys:
            name = add_planned(name)
            example[name] = outputs[name]

        # Split up into time-index features
        split_example = {}
        for name in dataset.action_feature_names:
            actions = example[name]
            for t in range(actions.shape[0]):
                action = actions[t]
                feature_name = ("%d/" + name) % t
                split_example[feature_name] = action

        for name in dataset.state_feature_names:
            states = example[name]
            for t in range(states.shape[0]):
                state = states[t]
                feature_name = ("%d/" + name) % t
                split_example[feature_name] = state

        for name in fwd_model.states_keys:
            name = add_planned(name)
            states = example[name]
            for t in range(states.shape[0]):
                state = states[t]
                feature_name = ("%d/" + name) % t
                split_example[feature_name] = state

        for name in dataset.constant_feature_names:
            split_example[name] = inputs[name]

        split_example = dict([(k, split_example[k]) for k in sorted(list(split_example))])
        return split_example

    new_tf_dataset = tf_dataset.map(_add_model_predictions)
    new_tf_dataset = new_tf_dataset.map(_split_sequences)
    return new_tf_dataset


class ClassifierDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path], params: Dict):
        super(ClassifierDataset, self).__init__(dataset_dirs)

        self.labeling_params = params

        self.full_env_params = FullEnvParams.from_json(self.hparams['full_env_params'])

        self.actual_state_keys = self.hparams['actual_state_keys']
        self.planned_state_keys = self.hparams['planned_state_keys']

        self.action_feature_names = ['action']

        self.state_feature_names = [
            'time_idx',
            'traj_idx',
        ]

        for k in self.actual_state_keys:
            self.state_feature_names.append('{}'.format(k))

        for k in self.planned_state_keys:
            self.state_feature_names.append(add_planned(k))

        self.constant_feature_names = [
            'full_env/origin',
            'full_env/extent',
            'full_env/env',
            'full_env/res',
        ]

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):

        # @tf.function
        def _label_transitions(transition: dict):
            state_key = self.labeling_params['state_key']
            state_key_next = add_next(state_key)
            planned_state_key = add_planned(state_key)
            planned_state_key_next = add_next(planned_state_key)
            pre_transition_distance = tf.norm(transition[state_key] - transition[planned_state_key])
            post_transition_distance = tf.norm(transition[state_key_next] - transition[planned_state_key_next])

            pre_threshold = self.labeling_params['pre_close_threshold']
            post_threshold = self.labeling_params['post_close_threshold']

            pre_close = pre_transition_distance < pre_threshold
            post_close = post_transition_distance < post_threshold

            # You're not allowed to modify input arguments, so we create a new dict and copy everything
            new_transition = {}
            for k, v in transition.items():
                new_transition[k] = v
            new_transition['pre_dist'] = pre_transition_distance
            new_transition['post_dist'] = post_transition_distance
            new_transition['pre_close'] = pre_close

            new_transition['label'] = None  # yes this is necessary. You can't add a key to a dict inside a py_func conditionally
            if post_close:
                new_transition['label'] = tf.convert_to_tensor([1], dtype=tf.float32)
            else:
                new_transition['label'] = tf.convert_to_tensor([0], dtype=tf.float32)
            return new_transition

        # @tf.function
        def _filter_pre_far_transitions(transition):
            if self.labeling_params['discard_pre_far'] and not transition['pre_close']:
                return False
            return True

        def _convert_sequences_to_transitions(constant_data, state_like_sequences, action_like_sequences):
            return convert_sequences_to_transitions(constant_data=constant_data,
                                                    state_like_sequences=state_like_sequences,
                                                    action_like_sequences=action_like_sequences,
                                                    planned_state_keys=self.planned_state_keys)

        # At this point, the dataset consists of tuples (const_data, state_data, action_data)
        dataset = dataset.flat_map(_convert_sequences_to_transitions)
        dataset = dataset.map(_label_transitions)
        dataset = dataset.filter(_filter_pre_far_transitions)

        return dataset
