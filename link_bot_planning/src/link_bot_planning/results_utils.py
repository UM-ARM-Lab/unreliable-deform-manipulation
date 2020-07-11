from typing import Dict
import json
import pathlib


def labeling_params_from_planner_params(planner_params, fallback_labeling_params: Dict):
    classifier_model_dir = pathlib.Path(planner_params['classifier_model_dir'])
    classifier_hparams_filename = classifier_model_dir.parent / 'params.json'
    classifier_hparams = json.load(classifier_hparams_filename.open('r'))
    if 'labeling_params' in classifier_hparams:
        labeling_params = classifier_hparams['labeling_params']
    elif 'classifier_dataset_hparams' in classifier_hparams:
        labeling_params = classifier_hparams['classifier_dataset_hparams']['labeling_params']
    else:
        labeling_params = fallback_labeling_params
    return labeling_params
