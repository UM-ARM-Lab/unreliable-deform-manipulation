#!/bin/bash
./scripts/collect_dynamics_data.py gazebo dual collect_dynamics_params/car.json 16 fwd_model_data/car
./scripts/split_dataset.py fwd_model_data/car
./scripts/make_classifier_dataset.py fwd_model_data/car labeling_params/dual.json dy_trials/no_victor_longer_*/best_checkpoint classifier_data/car
cd ../link_bot_classifiers
./scripts/train_test_classifier.py train classifier_data/car/ hparams/classifier/dual.json -l car --epochs 1

