#!/bin/bash
./scripts/collect_dynamics_data.py gazebo dual collect_dynamics_params/car.json 2048 fwd_model_data/car2 && \
  ./scripts/split_dataset.py fwd_model_data/car2 && \
  ./scripts/make_classifier_dataset.py fwd_model_data/car2 labeling_params/dual.json dy_trials/no_victor_longer_*/best_checkpoint classifier_data/car2 && \
  cd ../link_bot_classifiers && \
  ./scripts/train_test_classifier.py train classifier_data/car2 hparams/classifier/dual.json -l car2 --epochs 10

