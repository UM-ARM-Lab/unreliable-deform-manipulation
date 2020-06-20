import json
import pathlib
from typing import Tuple, List, Dict

import tensorflow as tf

from link_bot_pycommon.get_scenario import get_scenario
from shape_completion_training.model.filepath_tools import load_trial
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.full_dynamics_nn import ObstacleNNWrapper
from state_space_dynamics.image_cond_dyn import ImageCondDynamicsWrapper
from state_space_dynamics.rigid_translation_model import RigidTranslationModel
from state_space_dynamics.unconstrained_dynamics_nn import UDNNWrapper


def load_generic_model(model_dir) -> [BaseDynamicsFunction, Tuple[str]]:
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :return: the model class, and a list of strings describing the model
    """
    # FIXME: remove batch_size=1 here? can I put it in base model?
    if isinstance(model_dir, list):
        fwd_model_dirs = [pathlib.Path(d) for d in model_dir]
        fwd_model = EnsembleDynamicsFunction(fwd_model_dirs, batch_size=1)
        model_path_info = list(fwd_model_dirs[0].parts[1:])
        model_path_info[-1] = model_path_info[-1][:-2]  # remove the "-$n" so it's "dir/ensemble" instead of "dir/ensemble-$n"
        return fwd_model, model_path_info
    else:
        _, hparams = load_trial(model_dir.absolute())
        scenario = get_scenario(hparams['dynamics_dataset_hparams']['scenario'])
        model_type = hparams['model_class']
        if model_type == 'rigid':
            return RigidTranslationModel(model_dir, batch_size=1, scenario=scenario), model_dir.parts[1:]
        elif model_type == 'SimpleNN':
            nn = UDNNWrapper(model_dir, batch_size=1, scenario=scenario)
            return nn, model_dir.parts[1:]
        elif model_type == 'ObstacleNN':
            nn = ObstacleNNWrapper(model_dir, batch_size=1, scenario=scenario)
            return nn, model_dir.parts[1:]
        elif model_type == 'ImageCondDyn':
            nn = ImageCondDynamicsWrapper(model_dir, batch_size=1, scenario=scenario)
            return nn, model_dir.parts[1:]
        else:
            raise NotImplementedError("invalid model type {}".format(model_type))


def get_model_info(model_dir: pathlib.Path) -> Tuple[str]:
    """
    Loads a model which exposes a unified model API (predict, dt, n_state, etc...)
    :param model_dir: directory which specifies which model should loaded (TF assumes latest checkpoint)
    :return: the model class, and a list of strings describing the model
    """
    model_hparams_file = model_dir / 'hparams.json'
    hparams = json.load(model_hparams_file.open('r'))
    model_type = hparams['type']
    if model_type == 'rigid':
        return model_dir.parts[1:]
    elif model_type == 'SimpleNN':
        return model_dir.parts[1:]
    elif model_type == 'ObstacleNN':
        return model_dir.parts[1:]
    else:
        raise NotImplementedError("invalid model type {}".format(model_type))


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
        _, hparams = load_trial(fwd_model_dirs[0].absolute())
        scenario = get_scenario(hparams['dynamics_dataset_hparams']['scenario'])
        super().__init__(fwd_model_dirs[0], batch_size, scenario)
        self.models, _ = load_ensemble(fwd_model_dirs)
        self.n_models = len(self.models)
        self.states_keys = self.models[0].states_keys
        self.dynamics_data_params = self.models[0].dynamics_data_params
        self.full_env_params = self.models[0].full_env_params

    def propagate_from_example(self, dataset_element):
        all_predictions = {}
        for fwd_model in self.models:
            prediction = fwd_model.propagate_from_example(dataset_element)
            for k, v in prediction.items():
                if k not in all_predictions:
                    all_predictions[k] = []
                all_predictions[k].append(v)
        all_stdevs = []
        total_prediction = {}
        for k, v in all_predictions.items():
            mean_prediction = tf.math.reduce_mean(tf.stack(v, axis=0), axis=0)
            all_stdevs.append(tf.math.reduce_std(tf.stack(v, axis=0), axis=0))
            total_prediction[k] = mean_prediction

        total_stdev = tf.reduce_sum(tf.stack(all_stdevs, axis=0), axis=0)
        total_prediction['stdev'] = total_stdev
        return total_prediction

    def propagate_differentiable(self,
                                 environment: Dict,
                                 start_states: Dict,
                                 actions: tf.Variable) -> List[Dict]:
        all_predictions = []
        for fwd_model in self.models:
            predictions = fwd_model.propagate_differentiable(environment=environment,
                                                             start_states=start_states,
                                                             actions=actions)
            all_predictions.append(predictions)

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
            total_stdev = tf.reduce_sum(tf.stack(all_stdevs, axis=0), axis=0, keepdims=True)
            merged_predictions['stdev'] = total_stdev
            ensemble_predictions.append(merged_predictions)

        return ensemble_predictions

    @tf.function
    def propagate_differentiable_batched(self,
                                         start_states: Dict,
                                         actions: tf.Variable) -> Dict:
        all_predictions = []
        for fwd_model in self.models:
            net_input = {k: start_states[k] for k in fwd_model.states_keys}
            net_input.update({k: actions[k] for k in fwd_model.action_keys})
            predictions = fwd_model.net(net_input, training=False)
            all_predictions.append(predictions)

        # restructure data to be one List of dicts, where each dict has all the states/keys of the original dicts, but averaged
        # and with an additional state/key for stdev
        ensemble_predictions = {state_key: [] for state_key in self.states_keys}
        ensemble_predictions['stdev'] = []

        T = int(next(iter(actions.values())).shape[1]) + 1
        for t in range(T):
            all_stdevs_t = []
            for state_key in self.states_keys:
                predictions_t_key = []
                for model_idx in range(self.n_models):
                    # [batch, N]
                    prediction = all_predictions[model_idx][state_key][:, t]
                    predictions_t_key.append(prediction)
                predictions_t_key = tf.stack(predictions_t_key, axis=1)
                mean_for_t_key = tf.math.reduce_mean(predictions_t_key, axis=1)
                stdev_for_t_key = tf.math.reduce_std(predictions_t_key, axis=1)
                stdev_for_t_key = tf.math.reduce_sum(stdev_for_t_key, axis=1)
                all_stdevs_t.append(stdev_for_t_key)

                ensemble_predictions[state_key].append(mean_for_t_key)

            all_stdevs_t = tf.stack(all_stdevs_t, axis=1)
            ensemble_predictions['stdev'].append(tf.math.reduce_sum(all_stdevs_t, axis=1, keepdims=True))

        for state_key in ensemble_predictions.keys():
            ensemble_predictions[state_key] = tf.stack(ensemble_predictions[state_key], axis=1)

        return ensemble_predictions
