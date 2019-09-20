import numpy as np


def flatten_plan(np_controls, np_duration_steps_int):
    # flatten and combine np_controls and durations
    np_controls_flat = []
    for control, duration in zip(np_controls, np_duration_steps_int):
        for i in range(duration):
            np_controls_flat.append(control)
    np_controls_flat = np.array(np_controls_flat)
    return np_controls_flat


def flatten_plan_dt(np_controls, np_durations, dt):
    # flatten and combine np_controls and durations
    np_duration_steps_int = (np_durations / dt).astype(np.int)
    np_controls_flat = []
    for control, duration in zip(np_controls, np_duration_steps_int):
        for i in range(duration):
            np_controls_flat.append(control)
    np_controls_flat = np.array(np_controls_flat)
    return np_controls_flat
