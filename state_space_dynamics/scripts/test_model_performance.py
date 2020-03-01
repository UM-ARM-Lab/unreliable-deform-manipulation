#!/usr/bin/env python
import argparse
import dataclasses_json
import time

import numpy as np
import tensorflow as tf

import state_space_dynamics
from link_bot_planning.params import LocalEnvParams, FullEnvParams

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    batch = 32
    n_state = 10
    n_action = 2
    T = 10
    local_H = 50
    local_W = 50
    full_H = 400
    full_W = 400
    n_points = int(n_state / 2)

    module = state_space_dynamics.get_model_module(args.model_type)
    hparams = {
        'dynamics_dataset_hparams': {
            'n_state': n_state,
            'n_action': n_action,
            'local_env_params': LocalEnvParams(local_H, local_W, 0.01).to_json(),
            'full_env_params': FullEnvParams(full_H, full_W, 0.01).to_json(),
            'dt': 1.0,
        },
        "model_class": "ObstacleNN",
        "n_action": n_action,
        "sequence_length": T,
        "n_points": n_points,
        "conv_filters": [
            [8, [5, 5]],
            [8, [5, 5]]
        ],
        "fc_layer_sizes": [256, 256],
        "residual": True,
        "mixed": True,
        "kernel_reg": 0.0,
        "bias_reg": 0.0,
        "activity_reg": 0.0
    }
    net = module.model(hparams=hparams)

    state = np.zeros((batch, n_state))
    actions = np.zeros((batch, T, n_action))
    resolution_s = np.ones((batch, T)) * 0.01
    full_envs = np.random.randn(batch, full_H, full_W)
    full_env_origins = np.zeros((batch, 2))

    # let the auto-tuned parameters warm up
    # this also causes the model to "build" which means we can count parameters after
    for i in range(2):
        module.wrapper.static_predict(net, full_envs, full_env_origins, resolution_s, state, actions)

    print("num params:", net.count_params())

    t0 = time.time()
    dts = []
    for i in range(50):
        t_start = time.time()
        module.wrapper.static_predict(net, full_envs, full_env_origins, resolution_s, state, actions)
        dt = time.time() - t_start
        dts.append(dt)

        print("{:6.4f}".format(dt))
        if time.time() - t0 > 25:
            break

    print("min {:6.4f}".format(np.min(dts)))
    print("max {:6.4f}".format(np.max(dts)))
    print("mean {:6.4f}".format(np.mean(dts)))
    print("median {:6.4f}".format(np.median(dts)))
    print("std {:6.4f}".format(np.std(dts)))


if __name__ == '__main__':
    main()
