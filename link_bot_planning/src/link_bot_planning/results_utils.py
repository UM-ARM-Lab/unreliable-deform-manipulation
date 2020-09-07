import json
from typing import Dict

from link_bot_pycommon.pycommon import paths_from_json


def labeling_params_from_planner_params(planner_params, fallback_labeling_params: Dict):
    classifier_model_dirs = paths_from_json(planner_params['classifier_model_dir'])
    representative_classifier_model_dir = classifier_model_dirs[0]
    classifier_hparams_filename = representative_classifier_model_dir.parent / 'params.json'
    classifier_hparams = json.load(classifier_hparams_filename.open('r'))
    if 'labeling_params' in classifier_hparams:
        labeling_params = classifier_hparams['labeling_params']
    elif 'classifier_dataset_hparams' in classifier_hparams:
        labeling_params = classifier_hparams['classifier_dataset_hparams']['labeling_params']
    else:
        labeling_params = fallback_labeling_params
    return labeling_params
