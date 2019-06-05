import argparse
import bisect
from time import time

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from tabulate import tabulate

from link_bot_pycommon.src.link_bot_pycommon import link_bot_pycommon as tpoc


def sdf_to_dict(sdf):
    sdf_dict = {}
    for row, col in np.ndindex(sdf.shape):
        d = sdf[row, col]
        if d not in sdf_dict:
            sdf_dict[d] = []
        sdf_dict[d].append((row, col))
    for k, v in sdf_dict.items():
        sdf_dict[k] = np.array(v)
    return sdf_dict


def select_data_at_constraint_boundary(states, constraints):
    data_at_boundary = []
    T = states.shape[1]
    for state_traj, constraint_traj in zip(states, constraints):
        for t_idx in range(T - 1):
            current_constraint = constraint_traj[t_idx]
            next_constraint = constraint_traj[t_idx + 1]
            # xor operator
            if bool(current_constraint) ^ bool(next_constraint):
                average_state = (state_traj[t_idx] + state_traj[t_idx + 1]) / 2
                data_at_boundary.append(average_state)
    data_at_boundary = np.array(data_at_boundary)
    return data_at_boundary


def nearby_sdf_lookup(sdf_dict, origin, res, threshold):
    sorted_distance_keys = sdf_dict['sorted_distance_keys']
    nearest_distance_key = bisect.bisect_left(sorted_distance_keys, threshold)
    left_idx = 0
    right_idx = 0
    left_idx_near = True
    right_idx_near = True
    nearby_distance_keys = [nearest_distance_key]
    while left_idx_near or right_idx_near:
        if left_idx_near and nearest_distance_key + left_idx < len(sorted_distance_keys) - 1:
            left_idx += 1
            left_d = sorted_distance_keys[nearest_distance_key - left_idx]
            if abs(left_d - threshold) > 0.005:
                left_idx_near = False
            else:
                nearby_distance_keys.append(nearest_distance_key - left_idx)
        if right_idx_near and nearest_distance_key + right_idx < len(sorted_distance_keys) - 1:
            right_idx += 1
            right_d = sorted_distance_keys[nearest_distance_key + right_idx]
            if abs(right_d - threshold) > 0.005:
                right_idx_near = False
            else:
                nearby_distance_keys.append(nearest_distance_key + right_idx)

    indeces = np.concatenate([sdf_dict[sorted_distance_keys[k]] for k in nearby_distance_keys])
    points = tpoc.sdf_indeces_to_point(indeces, resolution=res, origin=origin)
    return points


correspondence_cache = {}


def func(sdf_dict, origin, res, data_at_constraint_boundary, params):
    global correspondence_cache

    # iterate over the data and find the data points which are on the boundary of collision, and take their average
    Rk = np.concatenate(([1], params[0:2]))
    data_at_constraint_boundary = np.copy(data_at_constraint_boundary)
    data_at_constraint_boundary = np.transpose(data_at_constraint_boundary.reshape(108, 3, 2), [0, 2, 1])
    data_at_constraint_boundary[:, :, 1:] -= data_at_constraint_boundary[:, :, :1]
    transformed_data = data_at_constraint_boundary @ Rk

    # The third parameter represents the distance from the edge of the object to the obstacle,
    # and then there is a required boundary of 10cm
    sdf_threshold = params[2] + 0.1
    if sdf_threshold in correspondence_cache:
        sdf_points_at_threshold, kd_tree = correspondence_cache[sdf_threshold]
        _, correspondence_guess = kd_tree.query(transformed_data)
    else:
        sdf_points_at_threshold = nearby_sdf_lookup(sdf_dict, origin, res, sdf_threshold)
        kd_tree = cKDTree(data=sdf_points_at_threshold)
        correspondence_cache[sdf_threshold] = (sdf_points_at_threshold, kd_tree)
        _, correspondence_guess = kd_tree.query(transformed_data)

    corresponding_sdf_points = sdf_points_at_threshold[correspondence_guess]

    error = np.linalg.norm(corresponding_sdf_points - transformed_data, axis=1)
    return error, corresponding_sdf_points


def attempt_minimize(args, sdf_dict, origin, res, data_at_constraint_boundary, out_params=[]):
    def _func(params):
        errors, corresponding_sdf_points = func(sdf_dict, origin, res, data_at_constraint_boundary, params)
        out_params.append(corresponding_sdf_points)
        return errors

    # these three numbers represent a linear combination of the points in the object
    initial_a = np.random.randn(2)
    initial_object_radius = np.random.uniform(0.0, 0.20, size=1)
    # initial_a = [0, 1]
    # initial_object_radius = [0.1]
    initial_params = np.concatenate((initial_a, initial_object_radius))
    if args.method == 'lm':
        sol = least_squares(_func, x0=initial_params, method='lm', loss='linear')
    elif args.method == 'trf':
        bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
        sol = least_squares(_func, x0=initial_params, bounds=bounds, method='trf', loss='huber')
    else:
        raise ValueError("invalid method " + args.method)
    return sol


def setup(args):
    data = np.load(args.data)
    states = data['states']
    constraints = data['constraints']

    sdf, _, sdf_resolution, sdf_origin = tpoc.load_sdf(args.sdf)

    # convert the sdf into a dict of {distance: [(row, col), (row, col), ...], distance:, [...], ...}
    sdf_dict = sdf_to_dict(sdf)
    # FIXME: sneaky add of this key to keep the data together
    sorted_distance_keys = sorted(sdf_dict.keys())
    sdf_dict['sorted_distance_keys'] = sorted_distance_keys

    data_at_constraint_boundary = select_data_at_constraint_boundary(states, constraints)

    return sdf, sdf_resolution, sdf_origin, states, constraints, sdf_dict, data_at_constraint_boundary


def solve_once(args):
    sdf, sdf_resolution, sdf_origin, states, constraints, sdf_dict, data_at_constraint_boundary = setup(args)

    maximum_iterations = 200
    t0 = time()
    sol = None
    mean_error = np.inf
    sdf_points_at_threshold = None
    success = False
    for attempt in range(maximum_iterations):
        out_params = []
        sol = attempt_minimize(args, sdf_dict, sdf_origin, sdf_resolution, data_at_constraint_boundary, out_params)
        mean_error = np.mean(sol.fun)
        sdf_points_at_threshold = out_params[0]
        if mean_error < args.success_threshold:
            success = True
            break

    dt = time() - t0

    if args.plot:
        # show all location of the tail overlayed on the SDF
        plt.style.use('slides')
        plt.figure()
        subsample = 10
        plt.imshow(np.flipud(sdf.T), extent=[-5, 5, -5, 5])
        candidate_xs = sdf_points_at_threshold[::subsample, 0]
        candidate_ys = sdf_points_at_threshold[::subsample, 1]
        plt.scatter(candidate_xs, candidate_ys, c='y', s=1, alpha=0.8, label='points in sdf at constraint boundary')
        plt.scatter(data_at_constraint_boundary[:, 4], data_at_constraint_boundary[:, 5], c='r', s=10,
                    label='head at constraint boundary')
        plt.legend()
        plt.axis("equal")
        plt.show()

    return dt, attempt, success, sol.x, mean_error


def solve_once_main(args):
    np.random.seed(args.seed)
    dt, attempts, success, params, mean_error = solve_once(args)
    color = Fore.GREEN if success else Fore.RED
    print(color + "mean error: {:6.4f}m".format(mean_error) + Fore.RESET)
    print("parameters: {}".format(params))
    print("attempts: {}".format(attempts))
    print("solve time: {:6.4f}s".format(dt))


def evaluate(args):
    np.random.seed(args.seed)
    dts = np.ndarray(args.n_runs)
    successes = 0
    errors = np.ndarray(args.n_runs)
    attempts = np.ndarray(args.n_runs)
    for i in range(args.n_runs):
        dt, attempt, success, _, error, = solve_once(args)
        print('.', end='')
        dts[i] = dt
        successes += 1 if success else 0
        errors[i] = error
        attempts[i] = attempt

    headers = ['metric', 'min', 'max', 'mean', 'median']
    metrics = [
        ['error (m)', np.min(errors), np.max(errors), np.mean(errors), np.median(errors)],
        ['attempts', np.min(attempts), np.max(attempts), np.mean(attempts), np.median(attempts)],
        ['time (s)', np.min(dts), np.max(dts), np.mean(dts), np.median(dts)],
    ]
    table = tabulate(metrics, headers=headers, tablefmt='github', floatfmt='6.3f')
    print()
    print(table)
    print('successes: {}/{}'.format(successes, args.n_runs))


def main():
    np.set_printoptions(suppress=True, linewidth=200, precision=2)

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('sdf')
    parser.add_argument('--method', choices=['lm', 'trf'], default='lm')
    parser.add_argument('--success-threshold', type=float, default=0.0105)

    subparsers = parser.add_subparsers()
    solve_once_parser = subparsers.add_parser('solve_once')
    solve_once_parser.add_argument('--plot', action='store_true')
    solve_once_parser.add_argument('--seed', type=int, default=None)
    solve_once_parser.set_defaults(func=solve_once_main)

    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('--n-runs', type=int, default=250)
    evaluate_parser.add_argument('--plot', action='store_true')
    evaluate_parser.add_argument('--seed', type=int, default=1)
    evaluate_parser.set_defaults(func=evaluate)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
