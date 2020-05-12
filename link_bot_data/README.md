To view the contents of a dataset, you can run 

    ./scripts/debug_tfrecord.py fwd_model_data/my_dataset_name

This should print something like

```bash
fwd_model_data/tf2_rope_obs_1587694454_511f2c7e3c_5120/train/traj_3968_to_4095.tfrecords
0/action: <BYTES>,
0/gripper: <BYTES>,
0/link_bot: <BYTES>,
0/time_idx: <BYTES>,
0/traj_idx: <BYTES>,
1/action: <BYTES>,
1/gripper: <BYTES>,
1/link_bot: <BYTES>,
1/time_idx: <BYTES>,
1/traj_idx: <BYTES>,
2/action: <BYTES>,
2/gripper: <BYTES>,
2/link_bot: <BYTES>,
2/time_idx: <BYTES>,
2/traj_idx: <BYTES>,
...
7/traj_idx: <BYTES>,
8/action: <BYTES>,
8/gripper: <BYTES>,
8/link_bot: <BYTES>,
8/time_idx: <BYTES>,
8/traj_idx: <BYTES>,
9/action: <BYTES>,
9/gripper: <BYTES>,
9/link_bot: <BYTES>,
9/time_idx: <BYTES>,
9/traj_idx: <BYTES>,
full_env/env: <BYTES>,
full_env/extent: <BYTES>,
full_env/origin: <BYTES>,
full_env/res: <BYTES>,
```

The `link_bot` and `gripper` are the states, and you can add/remove whatever states you want.

Some of the hparams of the dataset are json serialized dictionaries which can be saved/loaded with the `dataclasses_json` library

For the tensorboard folder, use

    tensorboard --logdir=log_data