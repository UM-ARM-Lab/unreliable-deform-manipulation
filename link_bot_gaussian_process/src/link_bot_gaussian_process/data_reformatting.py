import numpy as np


def make_relative_to_head(states):
    states_relative_to_head = np.copy(states)
    assert states.ndim > 1
    # flatten then reshape to make this function work for N-dim arrays
    states_relative_to_head = states_relative_to_head.reshape(-1, states.shape[-1])
    for s in states_relative_to_head:
        s[0] -= s[4]
        s[1] -= s[5]
        s[2] -= s[4]
        s[3] -= s[5]
        s[4] = 0
        s[5] = 0
    return states_relative_to_head.reshape(states.shape)


def format_forward_data(data, traj_idx_start=0, traj_idx_end=-1):
    """
    input to the forward model is a position of each point on the rope relative to the head,
    concatentated with the control input
    """
    states = data['states'][traj_idx_start:traj_idx_end]
    actions = data['actions'][traj_idx_start:traj_idx_end]

    # compute the delta in world frame
    delta = states[:, 1:] - states[:, :-1]
    # make input data more zero centered by making things relative to the head
    states_relative_to_head = make_relative_to_head(states)

    # flatten & concatenate data into various useful formats
    states_flat = states_relative_to_head[:, :-1].reshape(-1, 6)
    actions_flat = actions[:, :, :].reshape(-1, 2)
    combined_x = np.concatenate((states_flat, actions_flat), axis=1)
    delta_flat = delta.reshape(-1, 6)

    return states_flat, delta_flat, actions_flat, combined_x, states[:, :-1], actions


def format_inverse_data(data, traj_idx_start=0, traj_idx_end=-1, take_every=1):
    states = data['states'][traj_idx_start:traj_idx_end]
    actions = data['actions'][traj_idx_start:traj_idx_end]

    delta = states[:, take_every:] - states[:, :-take_every]

    delta_flat = delta.reshape(-1, 6)
    if take_every == 1:
        actions_flat = actions.reshape(-1, 2)
    else:
        actions_flat = actions[:, :(-take_every + 1)].reshape(-1, 2)
    mag_flat = np.linalg.norm(actions_flat, axis=1).reshape(-1, 1)
    actions_flat_scaled = actions_flat / np.linalg.norm(actions_flat, axis=1).reshape(-1, 1)
    # the action representation here is cos(theta), sin(theta), magnitude
    # I think this is better than predicting just components or mag/theta
    # because theta is discontinuous and GPs assume smoothness
    angle_actions_flat = np.concatenate((actions_flat_scaled, mag_flat), axis=1)

    return delta_flat, angle_actions_flat
