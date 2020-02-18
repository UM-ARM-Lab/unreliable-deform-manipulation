from typing import Optional, List, Tuple

import numpy as np
from colorama import Fore


def indeces_to_point(rowcols, resolution, origin):
    return (rowcols - origin) * resolution


def idx_to_point(row: int,
                 col: int,
                 resolution: np.ndarray,
                 origin: np.ndarray):
    y = (row - origin[0]) * resolution[0]
    x = (col - origin[1]) * resolution[1]
    return np.array([x, y])

def bounds_from_env_size(w_cols, h_rows, new_origin, resolution, origin):
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
    env_00_x = center_x - w_cols / 2 * res
    env_00_y = center_y - h_rows / 2 * res
    return np.array([int(-env_00_x / res), int(-env_00_y / res)])


def compute_extent(rows, cols, resolution, origin):
    """
    :param rows: scalar
    :param cols: scalar
    :param resolution: [2]
    :param origin: [2]
    :return:
    """
    xmin, ymin = idx_to_point(0, 0, resolution, origin)
    xmax, ymax = idx_to_point(rows, cols, resolution, origin)
    return np.array([xmin, xmax, ymin, ymax], dtype=np.float32)


def point_to_idx(x, y, resolution, origin):
    col = int(x / resolution[1] + origin[1])
    row = int(y / resolution[0] + origin[0])
    return row, col


class OccupancyData:

    def __init__(self,
                 data: np.ndarray,
                 resolution: np.ndarray,
                 origin: np.ndarray):
        """

        :param data:
        :param resolution: should be [res_y, res_x]
        :param origin:
        """
        self.data = data.astype(np.float32)
        self.resolution = resolution.astype(np.float32)
        # Origin means the indeces (row/col) of the world point (0, 0)
        self.origin = origin.astype(np.float32)
        self.extent = compute_extent(self.data.shape[0], self.data.shape[1], resolution, origin)
        # NOTE: when displaying an SDF as an image, matplotlib assumes rows increase going down,
        #  but rows correspond to y which increases going up
        self.image = np.flipud(self.data)


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


def make_rope_images(sdf_data, rope_configurations):
    rope_configurations = np.atleast_2d(rope_configurations)
    m, N = rope_configurations.shape
    n_rope_points = int(N / 2)
    rope_images = np.zeros([m, sdf_data.sdf.shape[0], sdf_data.sdf.shape[1], n_rope_points])
    for i in range(m):
        for j in range(n_rope_points):
            px = rope_configurations[i, 2 * j]
            py = rope_configurations[i, 2 * j + 1]
            row, col = point_to_idx(px, py, sdf_data.resolution, sdf_data.origin)
            rope_images[i, row, col, j] = 1
    return rope_images


def inflate(local_env: OccupancyData, radius_m: float):
    assert radius_m >= 0
    if radius_m == 0:
        return local_env

    inflated = local_env
    radius = int(radius_m / local_env.resolution[0])

    for i, j in np.ndindex(local_env.data.shape):
        try:
            if local_env.data[i, j] == 1:
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        inflated.data[i + di, j + dj] = 1
        except IndexError:
            pass

    return inflated
