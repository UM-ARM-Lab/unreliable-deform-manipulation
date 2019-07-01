import sdf_tools
from sdf_tools.srv import ComputeSDFRequest

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


