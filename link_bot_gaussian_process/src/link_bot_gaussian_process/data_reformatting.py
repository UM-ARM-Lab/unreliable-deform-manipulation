import numpy as np


def format_forward_data(data, traj_idx_start=0, traj_idx_end=-1, take_every=1):
    states = data['states'][traj_idx_start:traj_idx_end]
    for traj_idx in range(states.shape[0]):
        states[traj_idx, :, 0] -= states[traj_idx, 0, 4]
        states[traj_idx, :, 1] -= states[traj_idx, 0, 5]
        states[traj_idx, :, 2] -= states[traj_idx, 0, 4]
        states[traj_idx, :, 3] -= states[traj_idx, 0, 5]
        states[traj_idx, :, 4] -= states[traj_idx, 0, 4]
        states[traj_idx, :, 5] -= states[traj_idx, 0, 5]
    actions = data['actions'][traj_idx_start:traj_idx_end]
    state_dim = states.shape[2]
    action_dim = actions.shape[2]
    s = states[:, :-take_every, :]
    s_next = states[:, take_every:, :]
    s_flat = s.reshape(-1, state_dim)
    s_next_flat = s_next.reshape(-1, state_dim)
    if take_every == 1:
        u_flat = actions[:, :, :].reshape(-1, action_dim)
    else:
        u_flat = actions[:, :-(take_every - 1), :].reshape(-1, action_dim)

    combined_x = np.concatenate((s_flat, u_flat), axis=1)
    # make data more zero centered by making things relative to the head
    # input is [x_tail - x_head, y_tail - y_head, x_mid - x_head, y_mid - y_head, 0, 0, vx_head, vy_head]
    # output is [delta x_tail, delta y_tail, delt x_mid, delta y_mid, delta x_head, delta_y_head]
    x_flat = s_flat
    # predict the delta only
    y_flat = s_next_flat - s_flat
    # when doing rollouts, first you have to take the
    x_trajs = s
    u_trajs = actions
    return x_flat, y_flat, u_flat, combined_x, x_trajs, u_trajs


def format_inverse_data(data, traj_idx_start=0, traj_idx_end=-1, take_every=1):
    states = data['states'][traj_idx_start:traj_idx_end]
    for traj_idx in range(states.shape[0]):
        states[traj_idx, :, 0] -= states[traj_idx, 0, 4]
        states[traj_idx, :, 1] -= states[traj_idx, 0, 5]
        states[traj_idx, :, 2] -= states[traj_idx, 0, 4]
        states[traj_idx, :, 3] -= states[traj_idx, 0, 5]
        states[traj_idx, :, 4] -= states[traj_idx, 0, 4]
        states[traj_idx, :, 5] -= states[traj_idx, 0, 5]
    actions = data['actions'][traj_idx_start:traj_idx_end]
    state_dim = states.shape[2]
    action_dim = actions.shape[2]
    s = states[:, :-take_every, :]
    s_next = states[:, take_every:, :]
    s_flat = s.reshape(-1, state_dim)
    s_next_flat = s_next.reshape(-1, state_dim)
    if take_every == 1:
        u_flat = actions[:, :, :].reshape(-1, action_dim)
    else:
        u_flat = actions[:, :-(take_every - 1), :].reshape(-1, action_dim)

    combined_x = np.concatenate((s_flat, s_next_flat - s_flat), axis=1)
    return combined_x, u_flat
