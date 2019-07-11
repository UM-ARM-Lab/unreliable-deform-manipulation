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


def format_forward_data_gz(data, traj_idx_start=0, traj_idx_end=-1):
    """
    input to the forward model is a position of each point on the rope relative to the head,
    concatentated with the control input
    """
    states = data['states'][traj_idx_start:traj_idx_end]
    actions = data['actions'][traj_idx_start:traj_idx_end]

    # compute the delta
    delta = states[:, 1:] - states[:, :-1]
    # make input data more zero centered by making things relative to the head
    states_relative_to_head = make_relative_to_head(states)

    # flatten & concatenate data into various useful formats
    states_flat = states_relative_to_head[:, :-1].reshape(-1, 6)
    actions_flat = actions[:, :, :].reshape(-1, 2)
    combined_x = np.concatenate((states_flat, actions_flat), axis=1)
    delta_flat = delta.reshape(-1, 6)

    return states_flat, delta_flat, actions_flat, combined_x, states[:, :-1], actions


def format_inverse_data_gz(data, traj_idx_start=0, traj_idx_end=-1, examples_per_traj=50):
    """ this assumes trajectories have one constants control input """
    states = data['states'][traj_idx_start:traj_idx_end]
    actions = data['actions'][traj_idx_start:traj_idx_end]

    n_traj, max_n_steps, n_state = states.shape
    start_indeces = np.random.randint(1, max_n_steps, size=(n_traj * examples_per_traj))
    end_indeces = np.random.randint(1, max_n_steps, size=(n_traj * examples_per_traj))
    for i in np.argwhere(start_indeces > end_indeces):
        # https://stackoverflow.com/questions/14836228/is-there-a-standardized-method-to-swap-two-variables-in-python
        # yes, this is correct.
        start_indeces[i], end_indeces[i] = end_indeces[i], start_indeces[i]
    for i in np.argwhere(start_indeces == end_indeces):
        if end_indeces[i] == max_n_steps - 1:
            start_indeces[i] -= 1
        else:
            end_indeces[i] += 1

    traj_indeces = np.repeat(np.arange(n_traj), examples_per_traj)
    delta = states[traj_indeces, end_indeces] - states[traj_indeces, start_indeces]

    delta_flat = delta.reshape(-1, 6)
    head_delta = np.linalg.norm(delta_flat[:, 4:6], axis=1, keepdims=True)
    actions_flat = actions[traj_indeces, 0]
    num_steps_flat = (end_indeces - start_indeces).reshape(-1, 1)
    mag_flat = np.linalg.norm(actions_flat, axis=1).reshape(-1, 1)
    actions_flat_scaled = actions_flat / np.linalg.norm(actions_flat, axis=1).reshape(-1, 1)
    # the action representation here is cos(theta), sin(theta), magnitude
    # I think this is better than predicting just components or mag/theta
    # because theta is discontinuous and GPs assume smoothness
    x = np.concatenate((delta_flat, head_delta), axis=1)
    y = np.concatenate((actions_flat_scaled, mag_flat, num_steps_flat), axis=1)

    return x, y


def format_forward_data(data, traj_idx_start=0, traj_idx_end=-1):
    """
    input to the forward model is a position of each point on the rope relative to the head,
    concatentated with the control input
    """
    states = data['states'][traj_idx_start:traj_idx_end]
    actions = data['actions'][traj_idx_start:traj_idx_end]

    # compute the delta
    delta = states[:, 1:] - states[:, :-1]
    # make input data more zero centered by making things relative to the head
    states_relative_to_head = make_relative_to_head(states)

    # flatten & concatenate data into various useful formats
    states_flat = states_relative_to_head[:, :-1].reshape(-1, 6)
    actions_flat = actions[:, :, :].reshape(-1, 2)
    combined_x = np.concatenate((states_flat, actions_flat), axis=1)
    delta_flat = delta.reshape(-1, 6)

    return states_flat, delta_flat, actions_flat, combined_x, states[:, :-1], actions


def format_inverse_data(data, traj_idx_start=0, traj_idx_end=-1, examples_per_traj=50):
    """ this assumes trajectories have one constants control input """
    states = data['states'][traj_idx_start:traj_idx_end]
    actions = data['actions'][traj_idx_start:traj_idx_end]

    n_traj, max_n_steps, n_state = states.shape
    start_indeces = np.random.randint(1, max_n_steps, size=(n_traj * examples_per_traj))
    end_indeces = np.random.randint(1, max_n_steps, size=(n_traj * examples_per_traj))
    for i in np.argwhere(start_indeces > end_indeces):
        # https://stackoverflow.com/questions/14836228/is-there-a-standardized-method-to-swap-two-variables-in-python
        # yes, this is correct.
        start_indeces[i], end_indeces[i] = end_indeces[i], start_indeces[i]
    for i in np.argwhere(start_indeces == end_indeces):
        if end_indeces[i] == max_n_steps - 1:
            start_indeces[i] -= 1
        else:
            end_indeces[i] += 1

    traj_indeces = np.repeat(np.arange(n_traj), examples_per_traj)
    delta = states[traj_indeces, end_indeces] - states[traj_indeces, start_indeces]

    delta_flat = delta.reshape(-1, 6)
    head_delta = np.linalg.norm(delta_flat[:, 4:6], axis=1, keepdims=True)
    actions_flat = actions[traj_indeces, 0]
    num_steps_flat = (end_indeces - start_indeces).reshape(-1, 1)
    mag_flat = np.linalg.norm(actions_flat, axis=1).reshape(-1, 1)
    actions_flat_scaled = actions_flat / np.linalg.norm(actions_flat, axis=1).reshape(-1, 1)
    # the action representation here is cos(theta), sin(theta), magnitude
    # I think this is better than predicting just components or mag/theta
    # because theta is discontinuous and GPs assume smoothness
    x = np.concatenate((delta_flat, head_delta), axis=1)
    y = np.concatenate((actions_flat_scaled, mag_flat, num_steps_flat), axis=1)

    return x, y
