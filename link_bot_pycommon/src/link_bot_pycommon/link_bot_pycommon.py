#!/usr/bin/env python

import numpy as np


def sdf_indeces_to_point(rowcols, resolution, origin):
    return (rowcols - origin) * resolution


def sdf_idx_to_point(row, col, resolution, sdf_origin):
    x = (col - sdf_origin[0, 0]) * resolution[0, 0]
    y = (row - sdf_origin[1, 0]) * resolution[1, 0]
    return np.array([[y], [x]])


def point_to_sdf_idx(x, y, resolution, sdf_origin):
    row = int(x / resolution[0] + sdf_origin[0])
    col = int(y / resolution[1] + sdf_origin[1])
    return row, col


def yaw_diff(a, b):
    diff = a - b
    greater_indeces = np.argwhere(diff > np.pi)
    diff[greater_indeces] = diff[greater_indeces] - 2 * np.pi
    less_indeces = np.argwhere(diff < -np.pi)
    diff[less_indeces] = diff[less_indeces] + 2 * np.pi
    return diff


def load_sdf(filename):
    npz = np.load(filename)
    sdf = npz['sdf']
    grad = npz['sdf_gradient']
    res = npz['sdf_resolution'].reshape(2)
    origin = np.array(sdf.shape, dtype=np.int32).reshape(2) // 2
    return sdf, grad, res, origin


def state_cost(s, goal):
    return np.linalg.norm(s[0, 0:2] - goal[0, 0:2])
