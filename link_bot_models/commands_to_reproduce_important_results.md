# Test the original M2 model on the data it was trained on

    ./scripts/linear_model_runner.py eval ../link_bot_teleop/data/250_50_random3.npy ./log_data/new_init_M2/February_07_14\:10\:44__96b4f633b3/nn.ckpt-10000

# Train the full linear constraint model

    ./scripts/linear_constraint_model_runner.py train ../link_bot_teleop/data/250_50_collision.npz -e 2000 ../link_bot_teleop/data/sdf.npz

# perfect constraint prediction

    ./scripts/constraint_model_runner.py train ../link_bot_teleop/data/250_50_sdf_based_collision.npz ../link_bot_teleop/data/obstacles_1_sdf.npz  -e 0
