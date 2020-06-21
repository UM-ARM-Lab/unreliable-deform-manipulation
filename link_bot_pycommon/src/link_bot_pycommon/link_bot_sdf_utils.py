from typing import Optional, List, Tuple, Dict

import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from mps_shape_completion_msgs.msg import OccupancyStamped
from std_msgs.msg import MultiArrayDimension, Float32MultiArray


def indeces_to_point(rowcols, resolution, origin):
    return (rowcols - origin) * resolution


def idx_to_point_3d(row: int,
                    col: int,
                    channel: int,
                    resolution: float,
                    origin: np.ndarray):
    y = (row - origin[0]) * resolution
    x = (col - origin[1]) * resolution
    z = (channel - origin[2]) * resolution
    return np.array([x, y, z])


def idx_to_point(row: int,
                 col: int,
                 resolution: float,
                 origin: np.ndarray):
    y = (row - origin[0]) * resolution
    x = (col - origin[1]) * resolution
    return np.array([x, y])


def bounds_from_env_size(w_cols: int,
                         h_rows: int,
                         new_origin: np.ndarray,
                         resolution: float,
                         origin: np.ndarray):
    # NOTE: assumes centered?
    xmin = -w_cols / 2 + new_origin[1]
    ymin = -h_rows / 2 + new_origin[0]
    xmax = w_cols / 2 + new_origin[1]
    ymax = h_rows / 2 + new_origin[0]
    rmin, cmin = point_to_idx(xmin, ymin, resolution, origin)
    rmax, cmax = point_to_idx(xmax, ymax, resolution, origin)
    return [rmin, rmax, cmin, cmax], [xmin, xmax, ymin, ymax]


def center_point_to_origin_indices(h_rows: int,
                                   w_cols: int,
                                   center_x: float,
                                   center_y: float,
                                   res: float):
    env_origin_x = center_x - w_cols / 2 * res
    env_origin_y = center_y - h_rows / 2 * res
    return np.array([int(-env_origin_x / res), int(-env_origin_y / res)])


def compute_extent_3d(rows: int,
                      cols: int,
                      channels: int,
                      resolution: float):
    """ assumes the origin is in the center """
    origin = np.array([rows // 2, cols // 2, channels // 2], np.int32)
    xmin, ymin, zmin = idx_to_point_3d(0, 0, 0, resolution, origin)
    xmax, ymax, zmax = idx_to_point_3d(rows, cols, channels, resolution, origin)
    return np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype=np.float32)


def extent_to_env_size(extent_3d):
    min_x, max_x, min_y, max_y, min_z, max_z = extent_3d
    env_h_m = abs(max_x - min_x)
    env_w_m = abs(max_y - min_y)
    env_c_m = abs(max_z - min_z)
    return env_h_m, env_w_m, env_c_m


def extent_to_env_shape(extent, res):
    env_h_m, env_w_m, env_c_m = extent_to_env_size(extent)
    env_h_rows = int(env_h_m / res)
    env_w_cols = int(env_w_m / res)
    env_c_channels = int(env_c_m / res)
    return env_h_rows, env_w_cols, env_c_channels


def extent_to_center(extent_3d):
    min_x, max_x, min_y, max_y, min_z, max_z = extent_3d
    cx = (max_x + min_x) / 2
    cy = (max_y + min_y) / 2
    cz = (max_z + min_z) / 2
    return cx, cy, cz


def environment_to_occupancy_msg(environment: Dict) -> OccupancyStamped:
    occupancy = Float32MultiArray()
    env = environment['env']
    # NOTE: The plugin assumes data is ordered [x,y,z] so tranpose here
    env = np.transpose(env, [1, 0, 2])
    occupancy.data = env.astype(np.float32).flatten().tolist()
    h_rows, w_cols, c_channels = env.shape
    occupancy.layout.dim.append(MultiArrayDimension(label='x', size=w_cols, stride=h_rows * w_cols * c_channels))
    occupancy.layout.dim.append(MultiArrayDimension(label='y', size=h_rows, stride=w_cols * c_channels))
    occupancy.layout.dim.append(MultiArrayDimension(label='z', size=c_channels, stride=c_channels))
    msg = OccupancyStamped()
    msg.occupancy = occupancy
    msg.scale = environment['res']
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'occupancy'
    return msg


def compute_extent(rows: int,
                   cols: int,
                   resolution: float,
                   origin: np.ndarray):
    """
    :param rows: scalar
    :param cols: scalar
    :param resolution: scalar
    :param origin: [2]
    :return:
    """
    xmin, ymin = idx_to_point(0, 0, resolution, origin)
    xmax, ymax = idx_to_point(rows, cols, resolution, origin)
    return np.array([xmin, xmax, ymin, ymax], dtype=np.float32)


def batch_point_to_idx_tf(x,
                          y,
                          resolution: float,
                          origin):
    col = tf.cast(x / resolution + origin[1], tf.int64)
    row = tf.cast(y / resolution + origin[0], tf.int64)
    return row, col


def point_to_idx_3d_in_env(x: float,
                           y: float,
                           z: float,
                           environment: Dict):
    return point_to_idx_3d(x, y, z, resolution=environment['res'], origin=environment['origin'])


def point_to_idx_3d(x: float,
                    y: float,
                    z: float,
                    resolution: float,
                    origin: np.ndarray):
    row = int(y / resolution + origin[0])
    col = int(x / resolution + origin[1])
    channel = int(z / resolution + origin[2])
    return row, col, channel


def point_to_idx(x: float,
                 y: float,
                 resolution: float,
                 origin: np.ndarray):
    col = int(x / resolution + origin[1])
    row = int(y / resolution + origin[0])
    return row, col


class OccupancyData:

    def __init__(self,
                 data: np.ndarray,
                 resolution: float,
                 origin: np.ndarray):
        """

        :param data:
        :param resolution: scalar, assuming square pixels
        :param origin:
        """
        self.data = data.astype(np.float32)
        self.resolution = resolution
        # Origin means the indeces (row/col) of the world point (0, 0)
        self.origin = origin.astype(np.float32)
        self.extent = compute_extent(self.data.shape[0], self.data.shape[1], resolution, origin)
        # NOTE: when displaying an 2d data as an image, matplotlib assumes rows increase going down,
        #  but rows correspond to y which increases going up
        self.image = np.flipud(self.data)

    def copy(self):
        copy = OccupancyData(data=np.copy(self.data),
                             resolution=self.resolution,
                             origin=np.copy(self.origin))
        return copy


def batch_occupancy_data(occupancy_data_s: List[OccupancyData]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_s = []
    res_s = []
    origin_s = []
    extent_s = []
    for data in occupancy_data_s:
        data_s.append(data.data)
        res_s.append(data.resolution)
        origin_s.append(data.origin)
        extent_s.append(data.extent)

    return np.array(data_s), np.array(res_s), np.array(origin_s), np.array(extent_s)


def unbatch_occupancy_data(data: np.ndarray,
                           resolution: np.ndarray,
                           origin: np.ndarray) -> List[OccupancyData]:
    batch_size = data.shape[0]
    datas = []
    for i in range(batch_size):
        occupancy_data = OccupancyData(data[i], resolution[i], origin[i])
        datas.append(occupancy_data)

    return datas


class SDF:

    def __init__(self,
                 sdf: np.ndarray,
                 gradient: Optional[np.ndarray],
                 resolution: np.ndarray,
                 origin: np.ndarray):
        self.sdf = sdf.astype(np.float32)
        if gradient is not None:
            self.gradient = gradient.astype(np.float32)
        self.resolution = resolution.astype(np.float32)
        # Origin means the indeces (row/col) of the world point (0, 0)
        self.origin = origin.astype(np.float32)
        self.extent = compute_extent(sdf.shape[0], sdf.shape[1], resolution, origin)
        # NOTE: when displaying an SDF as an image, matplotlib assumes rows increase going down,
        #  but rows correspond to y which increases going up
        self.image = np.flipud(sdf)

    def save(self, sdf_filename):
        np.savez(sdf_filename,
                 sdf=self.sdf,
                 sdf_gradient=self.gradient,
                 sdf_resolution=self.resolution,
                 sdf_origin=self.origin)

    @staticmethod
    def load(filename):
        with np.load(filename) as npz:
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


def env_from_occupancy_data(occupancy_data: OccupancyData):
    environment = {
        'full_env/env': occupancy_data.data,
        'full_env/origin': occupancy_data.origin,
        'full_env/res': occupancy_data.resolution,
        'full_env/extent': occupancy_data.extent,
    }
    return environment
