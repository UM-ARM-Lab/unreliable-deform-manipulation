import json
import pathlib
from typing import List, Dict

import numpy as np
import tensorflow as tf

from link_bot_planning.get_scenario import get_scenario
from link_bot_planning.model_utils import load_generic_model
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


def load_ensemble(fwd_model_dirs: List[pathlib.Path]):
    fwd_models = []
    model_info = None
    for fwd_model_dir in fwd_model_dirs:
        # just model_info overwrite and return the last one, they should all be identical anyways
        fwd_model, model_path_info = load_generic_model(fwd_model_dir)
        fwd_models.append(fwd_model)
    return fwd_models, model_info


class EnsembleDynamicsFunction(BaseDynamicsFunction):

    def __init__(self, fwd_model_dirs: List[pathlib.Path], batch_size: int):
        model_hparams_file = fwd_model_dirs[0] / 'hparams.json'
        hparams = json.load(model_hparams_file.open('r'))
        scenario = get_scenario(hparams['dynamics_dataset_hparams']['scenario'])
        super().__init__(fwd_model_dirs[0], batch_size, scenario)
        self.models, _ = load_ensemble(fwd_model_dirs)
        self.n_models = len(self.models)
        self.states_keys = self.models[0].states_keys

     # @tf.function
    def propagate_differentiable(self,
                                 full_env: np.ndarray,
                                 full_env_origin: np.ndarray,
                                 res: float,
                                 start_states: Dict[str, np.ndarray],
                                 actions: tf.Variable) -> List[Dict]:
        all_predictions = []
        for fwd_model in self.models:
            predictions = fwd_model.propagate_differentiable(full_env=full_env,
                                                             full_env_origin=full_env_origin,
                                                             res=res,
                                                             start_states=start_states,
                                                             actions=actions)
            all_predictions.append(predictions)
        del predictions

        # restructure data to be one List of dicts, where each dict has all the states/keys of the original dicts, but averaged
        # and with an additional state/key for stdev
        T = int(actions.shape[0]) + 1
        ensemble_predictions = []
        for t in range(T):
            merged_predictions = {}
            all_stdevs = []
            for state_key in self.states_keys:
                predictions_for_state_key = []
                for model_idx in range(self.n_models):
                    predictions_for_state_key.append(all_predictions[model_idx][t][state_key])
                predictions_for_state_key = tf.stack(predictions_for_state_key, axis=1)
                mean_for_key = tf.math.reduce_mean(predictions_for_state_key, axis=1)
                merged_predictions[state_key] = mean_for_key
                stdev_for_key = tf.math.reduce_sum(tf.math.reduce_std(predictions_for_state_key, axis=1), axis=0)
                all_stdevs.append(stdev_for_key)
            total_stdev = tf.reduce_sum(tf.stack(all_stdevs, axis=0), axis=0)
            merged_predictions['stdev'] = total_stdev
            ensemble_predictions.append(merged_predictions)

        return ensemble_predictions
