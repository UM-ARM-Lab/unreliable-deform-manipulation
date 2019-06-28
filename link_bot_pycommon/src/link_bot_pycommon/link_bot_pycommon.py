from __future__ import division

import struct
from enum import Enum

import numpy as np
from colorama import Fore

import sdf_tools
from sdf_tools.srv import ComputeSDFRequest


def make_rope_images(sdf_data, rope_configurations):
    rope_configurations = np.atleast_2d(rope_configurations)
    m, N = rope_configurations.shape
    n_rope_points = int(N / 2)
    rope_images = np.zeros([m, sdf_data.sdf.shape[0], sdf_data.sdf.shape[1], n_rope_points])
    for i in range(m):
        for j in range(n_rope_points):
            px = rope_configurations[i, 2 * j]
            py = rope_configurations[i, 2 * j + 1]
            row, col = point_to_sdf_idx(px, py, sdf_data.resolution, sdf_data.origin)
            rope_images[i, row, col, j] = 1
    return rope_images


def sdf_indeces_to_point(rowcols, resolution, origin):
    return (rowcols - origin) * resolution


def sdf_idx_to_point(row, col, resolution, origin):
    x = (col - origin[0]) * resolution[0]
    y = (row - origin[1]) * resolution[1]
    return np.array([y, x])


def sdf_bounds(sdf, resolution, origin):
    xmin, ymin = sdf_idx_to_point(0, 0, resolution, origin)
    xmax, ymax = sdf_idx_to_point(sdf.shape[0], sdf.shape[1], resolution, origin)
    return [xmin, xmax, ymin, ymax]


def point_to_sdf_idx(x, y, resolution, origin):
    row = int(x / resolution[0] + origin[0])
    col = int(y / resolution[1] + origin[1])
    return row, col


def yaw_diff(a, b):
    diff = a - b
    greater_indeces = np.argwhere(diff > np.pi)
    diff[greater_indeces] = diff[greater_indeces] - 2 * np.pi
    less_indeces = np.argwhere(diff < -np.pi)
    diff[less_indeces] = diff[less_indeces] + 2 * np.pi
    return diff


class SDF:

    def __init__(self, sdf, gradient, resolution, origin):
        self.sdf = sdf
        self.gradient = gradient
        self.resolution = resolution
        self.origin = origin
        self.extent = sdf_bounds(sdf, resolution, origin)
        self.image = np.flipud(sdf.T)

    def save(self, sdf_filename):
        np.savez(sdf_filename,
                 sdf=self.sdf,
                 sdf_gradient=self.gradient,
                 sdf_resolution=self.resolution,
                 sdf_origin=self.origin)

    @staticmethod
    def load(filename):
        npz = np.load(filename)
        sdf = npz['sdf']
        grad = npz['sdf_gradient']
        res = npz['sdf_resolution'].reshape(2)
        origin = npz['sdf_origin'].reshape(2)
        return SDF(sdf=sdf, gradient=grad, resolution=res, origin=origin)

    def __repr__(self):
        return "SDF: size={}x{} origin=({},{}) resolution=({},{})".format(self.sdf.shape[0],
                                                                          self.sdf.shape[1],
                                                                          self.origin[0],
                                                                          self.origin[1],
                                                                          self.resolution[0],
                                                                          self.resolution[1])


def request_sdf_data(get_sdf_service, res=0.05, robot_name='link_bot'):
    compute_sdf_request = ComputeSDFRequest()
    compute_sdf_request.center.x = 0
    compute_sdf_request.center.y = 0
    compute_sdf_request.request_new = True
    width = 10.0
    height = 10.0
    compute_sdf_request.resolution = res  # applies to both x/y dimensions
    compute_sdf_request.x_width = width
    compute_sdf_request.y_height = height
    compute_sdf_request.min_z = 0.01  # must be greater than zero or the ground plane will be included
    compute_sdf_request.max_z = 2  # must be higher than the highest obstacle
    compute_sdf_request.robot_name = robot_name

    x = get_sdf_service.call(compute_sdf_request)

    sdf = sdf_tools.SignedDistanceField()

    ints = struct.unpack('<' + 'B' * len(x.sdf.serialized_sdf), x.sdf.serialized_sdf)
    uncompressed_sdf_structure = sdf_tools.DecompressBytes(ints)

    sdf.DeserializeSelf(uncompressed_sdf_structure, 0, sdf_tools.DeserializeFixedSizePODFloat)
    np_sdf, np_gradient = sdf_tools.compute_gradient(sdf)
    np_resolution = np.array([res, res])
    np_origin = np.array([-height / res / 2, -width / res / 2])
    sdf_data = SDF(sdf=np_sdf, gradient=np_gradient, resolution=np_resolution, origin=np_origin)
    return sdf_data


def load_sdf(filename):
    npz = np.load(filename)
    sdf = npz['sdf']
    grad = npz['sdf_gradient']
    res = npz['sdf_resolution'].reshape(2)
    if 'sdf_origin' in npz:
        origin = npz['sdf_origin'].reshape(2)
    else:
        origin = np.array(sdf.shape, dtype=np.int32).reshape(2) // 2
        print(Fore.YELLOW + "WARNING: sdf npz file does not specify its origin, assume origin {}".format(origin) + Fore.RESET)
    return sdf, grad, res, origin


def state_cost(s, goal):
    return np.linalg.norm(s[0, 0:2] - goal[0, 0:2])


class ArgsEnum(Enum):

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return cls[s]
        except KeyError:
            raise ValueError()


def make_random_rope_configuration(extent, length=0.5):
    while True:
        theta_1 = np.random.uniform(-np.pi, np.pi)
        theta_2 = np.random.uniform(-np.pi, np.pi)
        # don't put the head so close to the edge that the tail could be off the map
        head_x = np.random.uniform(extent[0] + 2.0 * length, extent[1] - 2.0 * length)
        head_y = np.random.uniform(extent[2] + 2.0 * length, extent[3] - 2.0 * length)

        rope_configuration = np.zeros(6)
        rope_configuration[4] = head_x
        rope_configuration[5] = head_y
        rope_configuration[2] = rope_configuration[4] + np.cos(theta_1) * length
        rope_configuration[3] = rope_configuration[5] + np.sin(theta_1) * length
        rope_configuration[0] = rope_configuration[2] + np.cos(theta_2) * length
        rope_configuration[1] = rope_configuration[3] + np.sin(theta_2) * length

        if extent[0] < rope_configuration[0] < extent[1] and \
                extent[0] < rope_configuration[2] < extent[1] and \
                extent[0] < rope_configuration[4] < extent[1] and \
                extent[2] < rope_configuration[1] < extent[3] and \
                extent[2] < rope_configuration[3] < extent[3] and \
                extent[2] < rope_configuration[5] < extent[3]:
            break

    return rope_configuration


def make_rope_configuration(head_x, head_y, theta_1, theta_2, l=0.5):
    rope_configuration = np.zeros(6)
    rope_configuration[4] = head_x
    rope_configuration[5] = head_y
    rope_configuration[2] = rope_configuration[4] + np.cos(theta_1) * l
    rope_configuration[3] = rope_configuration[5] + np.sin(theta_1) * l
    rope_configuration[0] = rope_configuration[2] + np.cos(theta_2) * l
    rope_configuration[1] = rope_configuration[3] + np.sin(theta_2) * l
    return rope_configuration


def make_rope_configurations(head_xs, head_ys, theta_1s, theta_2s, l=0.5):
    n = head_xs.shape[0]
    rope_configurations = np.zeros((n, 6))
    rope_configurations[:, 4] = head_xs
    rope_configurations[:, 5] = head_ys
    rope_configurations[:, 2] = rope_configurations[:, 4] + np.cos(theta_1s) * l
    rope_configurations[:, 3] = rope_configurations[:, 5] + np.sin(theta_1s) * l
    rope_configurations[:, 0] = rope_configurations[:, 2] + np.cos(theta_2s) * l
    rope_configurations[:, 1] = rope_configurations[:, 3] + np.sin(theta_2s) * l
    return rope_configurations
