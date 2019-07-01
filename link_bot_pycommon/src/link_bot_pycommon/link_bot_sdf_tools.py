import struct

import numpy as np
from colorama import Fore

import sdf_tools
from link_bot_gazebo.srv import ComputeSDFRequest


def decompress_and_deserialize(sdf, gradient, msg):
    sdf_ints = struct.unpack('<' + 'B' * len(msg.sdf.serialized_sdf), msg.sdf.serialized_sdf)
    uncompressed_sdf = sdf_tools.DecompressBytes(sdf_ints)
    sdf.DeserializeSelf(uncompressed_sdf, 0, sdf_tools.DeserializeFixedSizePODFloat)

    gradient_ints = struct.unpack('<' + 'B' * len(msg.compressed_sdf_gradient), msg.compressed_sdf_gradient)
    uncompressed_gradient = sdf_tools.DecompressBytes(gradient_ints)
    gradient.DeserializeSelf(uncompressed_gradient, 0, sdf_tools.DeserializeFixedSizePODVecd)


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

    response = get_sdf_service.call(compute_sdf_request)

    sdf = sdf_tools.SignedDistanceField()
    sdf_ints = struct.unpack('<' + 'B' * len(response.sdf.serialized_sdf), response.sdf.serialized_sdf)
    uncompressed_sdf = sdf_tools.DecompressBytes(sdf_ints)
    sdf.DeserializeSelf(uncompressed_sdf, 0, sdf_tools.DeserializeFixedSizePODFloat)

    # sdf = sdf_tools.SignedDistanceField()
    # gradient = sdf_tools.VoxelGrid()
    # decompress_and_deserialize(sdf, gradient, response)

    np_sdf, np_gradient = sdf_tools.compute_gradient(sdf)
    np_resolution = np.array([res, res])
    np_origin = np.array([-height / res / 2, -width / res / 2])
    sdf_data = SDF(sdf=np_sdf, gradient=np_gradient, resolution=np_resolution, origin=np_origin)
    return sdf_data


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
