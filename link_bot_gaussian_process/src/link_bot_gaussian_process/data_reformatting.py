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


def format_forward_data_gz_tfrecords(x_rope_configurations, y_rope_configurations, actions):
    """
    input to the forward model is a position of each point on the rope relative to the head,
    concatenated with the control input
    """
    x_rope_configurations = x_rope_configurations.squeeze()
    y_rope_configurations = y_rope_configurations[:, 1:, :].squeeze()

    delta_flat = y_rope_configurations - x_rope_configurations

    # make input data more zero centered by making things relative to the head
    # NOTE: don't include the head point (0,0)
    relative_to_head = make_relative_to_head(x_rope_configurations)
    if relative_to_head.ndim == 2:
        relative_to_head = relative_to_head[:, :-2]
    else:
        relative_to_head = relative_to_head[:, :, :-2]

    actions = actions.squeeze()
    combined_x = np.concatenate((relative_to_head, actions), axis=-1)

    return combined_x, delta_flat


def format_forward_data_gz(args, dataset):
    """
    input to the forward model is a position of each point on the rope relative to the head,
    concatenated with the control input
    """

    # get all pairs of (states, next state) where neither has constraint label 1
    states_flat = []
    next_states = []
    actions = []
    for env in dataset.environments:
        rope_data = env.rope_data
        env_states = rope_data['rope_configurations']
        env_actions = rope_data['gripper1_target_velocities']
        env_mask_labels = rope_data[args.mask_label_type.name]

        for t in range(env_states.shape[0] - 1):
            if not env_mask_labels[t] and not env_mask_labels[t + 1]:
                states_flat.append(env_states[t])
                next_states.append(env_states[t + 1])
                actions.append(env_actions[t])
    states_flat = np.array(states_flat).astype(np.float64)
    next_states_flat = np.array(next_states).astype(np.float64)
    actions_flat = np.array(actions).astype(np.float64)

    # compute the delta
    delta_flat = next_states_flat - states_flat

    # make input data more zero centered by making things relative to the head
    states_relative_to_head_flat = make_relative_to_head(states_flat)

    # flatten & concatenate data into various useful formats
    # exclude the very last state because there is no corresponding action
    combined_x = np.concatenate((states_relative_to_head_flat, actions_flat), axis=1)
    y = delta_flat

    return states_relative_to_head_flat, y, actions_flat, combined_x, states_flat, actions


def format_inverse_data_gz(args, dataset):
    """ this assumes trajectories have one constant control input """

    # get all pairs of (states, next state) where neither has constraint label 1
    actions_flat = []
    delta_flat = []
    num_steps_flat = []
    j = 0
    for env in dataset.environments:
        rope_data = env.rope_data
        env_states = rope_data['rope_configurations']
        env_actions = rope_data['gripper1_target_velocities']
        env_mask_labels = rope_data[args.mask_label_type.name]

        for t_start in range(env_states.shape[0] - 2):
            if env_mask_labels[t_start] or env_mask_labels[t_start + 1]:
                continue
            # look forward until we find a mask label == 1 and take that length
            t_end = t_start + 1
            for t_end in range(t_start + 1, env_states.shape[0]):
                if env_mask_labels[t_end]:
                    break
            # then pick a random sequence starting at the beginning
            t_end = np.random.randint(t_start + 1, t_end)
            delta = env_states[t_end] - env_states[t_start]
            traj_action = env_actions[t_start]
            delta_flat.append(delta)
            actions_flat.append(traj_action)
            num_steps_flat.append(t_end - t_start)
            j += 1
    delta_flat = np.array(delta_flat).astype(np.float64)
    actions_flat = np.array(actions_flat).astype(np.float64)
    num_steps_flat = np.array(num_steps_flat).astype(np.float64)

    head_delta = np.linalg.norm(delta_flat[:, 4:6], axis=1, keepdims=True)
    actions_flat_normalized = actions_flat / np.linalg.norm(actions_flat, axis=1).reshape(-1, 1)
    mag_flat = np.linalg.norm(actions_flat, axis=1).reshape(-1, 1)
    num_steps_flat = num_steps_flat.reshape(-1, 1)

    # the action representation here is cos(theta), sin(theta), magnitude
    # I think this is better than predicting just components or mag/theta
    # because theta is discontinuous and GPs assume smoothness
    # FIXME: by including only the head delta as a feature we are super cheating
    x = np.concatenate((delta_flat, head_delta), axis=1)
    # y = np.concatenate((actions_flat_normalized, mag_flat, num_steps_flat), axis=1)
    # NOTE: asking the model to predict both magnitude and number of steps is ill-posed, there is no single correct solution
    # NOTE: in the new gazebo data, the target velocity changes over the course of a trajectory, so it doesn't make sense to
    #       predict the speed, only the direction of the action
    y = actions_flat_normalized

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
