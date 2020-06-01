import json
import pathlib

import tensorflow as tf

from link_bot_pycommon.get_scenario import get_scenario
from state_space_dynamics.full_dynamics_nn import FullDynamicsNN

hparams_filename = pathlib.Path("log_data/rope_full_dynamics3/March_26_14-34-54_4de27bda6b/hparams.json")
model_dir = pathlib.Path("log_data/rope_full_dynamics3/March_26_14-34-54_4de27bda6b/")
hparams = json.load(hparams_filename.open("r"))
batch_size = 1
scenario = get_scenario("link_bot")
net = FullDynamicsNN(hparams=hparams, batch_size=batch_size, scenario=scenario)
states_keys = net.states_keys
in_ckpt = tf.train.Checkpoint(net=net)
in_manager = tf.train.CheckpointManager(in_ckpt, model_dir, max_to_keep=1)
status = in_ckpt.restore(in_manager.latest_checkpoint)
assert in_manager.latest_checkpoint
status.assert_existing_objects_matched()

out_model_dir = pathlib.Path("log_data/rope_full_dynamics3_converted/March_26_14-34-54_4de27bda6b/")
out_ckpt = tf.train.Checkpoint(model=in_ckpt.net)
out_manager = tf.train.CheckpointManager(out_ckpt, out_model_dir, max_to_keep=1)
out_manager.save()

