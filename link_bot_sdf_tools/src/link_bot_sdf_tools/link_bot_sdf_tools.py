import struct

import numpy as np

import sdf_tools
# FIXME: imports not working?!
from link_bot_pycommon.link_bot_sdf_utils import SDF
from .srv import ComputeSDFRequest


def decompress_and_deserialize(sdf, gradient, msg):
    sdf_ints = struct.unpack('<' + 'B' * len(msg.sdf.serialized_sdf), msg.sdf.serialized_sdf)
    uncompressed_sdf = sdf_tools.DecompressBytes(sdf_ints)
    sdf.DeserializeSelf(uncompressed_sdf, 0, sdf_tools.DeserializeFixedSizePODFloat)

    gradient_ints = struct.unpack('<' + 'B' * len(msg.compressed_sdf_gradient), msg.compressed_sdf_gradient)
    uncompressed_gradient = sdf_tools.DecompressBytes(gradient_ints)
    gradient.DeserializeSelf(uncompressed_gradient, 0, sdf_tools.DeserializeFixedSizePODVecd)


def request_sdf_data(get_sdf_service, width=5.0, height=5.0, res=0.05, robot_name='link_bot'):
    compute_sdf_request = ComputeSDFRequest()
    compute_sdf_request.center.x = 0
    compute_sdf_request.center.y = 0
    compute_sdf_request.request_new = True
    compute_sdf_request.resolution = res  # applies to both x/y dimensions
    compute_sdf_request.x_width = width
    compute_sdf_request.y_height = height
    compute_sdf_request.min_z = 0.01  # must be greater than zero or the ground plane will be included
    compute_sdf_request.max_z = 0.5  # must be higher than the highest obstacle
    compute_sdf_request.robot_name = robot_name

    response = get_sdf_service(compute_sdf_request)

    sdf = sdf_tools.SignedDistanceField()
    sdf_ints = struct.unpack('<' + 'B' * len(response.sdf.serialized_sdf), response.sdf.serialized_sdf)
    uncompressed_sdf = sdf_tools.DecompressBytes(sdf_ints)
    sdf.DeserializeSelf(uncompressed_sdf, 0, sdf_tools.DeserializeFixedSizePODFloat)

    np_sdf, np_gradient = sdf_tools.compute_gradient(sdf)
    np_resolution = np.array([res, res])
    np_origin = np.array([height / res / 2, width / res / 2], np.int64)
    sdf_data = SDF(sdf=np_sdf, gradient=np_gradient, resolution=np_resolution, origin=np_origin)
    return sdf_data
