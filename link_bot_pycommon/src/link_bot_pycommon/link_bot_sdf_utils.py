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


def get_local_env_origins(center_points,
                          full_env_origins,
                          rows: int,
                          cols: int,
                          res: float):
    """
    NOTE: Assumes both local and full env have the same resolution
    :param center_points: [batch, 2] (x,y) meters
    :param full_env_origins: the full environment data origins
    :param rows: scalar, int
    :param cols: scalar, int
    :param res: scalar, meters
    :return: local env origins
    """
    # indeces of the heads of the ropes in the full env, with a batch dimension up front
    center_cols = (center_points[:, 0] / res + full_env_origins[:, 1]).astype(np.int64)
    center_rows = (center_points[:, 1] / res + full_env_origins[:, 0]).astype(np.int64)
    local_env_origins = full_env_origins - np.stack([center_rows, center_cols], axis=1) + np.array([rows // 2, cols // 2])
    local_env_origins = local_env_origins.astype(np.float32)
    return local_env_origins


def get_local_env_at_in(center_points: np.ndarray,
                        padded_full_envs: np.ndarray,
                        full_env_origins: np.ndarray,
                        padding: int,
                        rows: int,
                        cols: int,
                        res: float):
    """
    NOTE: Assumes both local and full env have the same resolution
    :param center_points: [batch, 2] (x,y) meters
    :param padded_full_envs: [batch, h, w] the full environment data
    :param full_env_origins: [batch, 2]
    :param padding: scalar
    :param rows: [batch]
    :param cols: [batch]
    :param res: scalar, meters
    :return: local envs
    """
    batch_size = int(center_points.shape[0])

    # indeces of the heads of the ropes in the full env, with a batch dimension up front
    center_cols = (center_points[:, 0] / res + full_env_origins[:, 1]).astype(np.int64)
    center_rows = (center_points[:, 1] / res + full_env_origins[:, 0]).astype(np.int64)
    delta_rows = np.tile(np.arange(-rows // 2, rows // 2), [batch_size, cols, 1]).transpose([0, 2, 1])
    delta_cols = np.tile(np.arange(-cols // 2, cols // 2), [batch_size, rows, 1])
    row_indeces = np.tile(center_rows, [cols, rows, 1]).T + delta_rows
    col_indeces = np.tile(center_cols, [cols, rows, 1]).T + delta_cols
    batch_indeces = np.tile(np.arange(0, batch_size), [cols, rows, 1]).transpose()
    local_env = padded_full_envs[batch_indeces, row_indeces + padding, col_indeces + padding]
    local_env = local_env.astype(np.float32)
    return local_env


def get_local_env_and_origin(head_point_t: np.ndarray,
                             full_env: np.ndarray,
                             full_env_origin: np.ndarray,
                             local_h_rows: int,
                             local_w_cols: int,
                             res: float):
    """
    :param local_h_rows: scalar
    :param local_w_cols: scalar
    :param head_point_t: [batch, 2]
    :param full_env_origin: [batch, 2]
    :param full_env: [batch, h, w]
    :param res: scalar
    :return:
    """
    padding = 200
    paddings = [[0, 0], [padding, padding], [padding, padding]]
    padded_full_envs = np.pad(full_env, paddings, 'constant', constant_values=0)
    local_env_origin = get_local_env_origins(head_point_t,
                                             full_env_origin,
                                             local_h_rows,
                                             local_w_cols,
                                             res)
    local_env = get_local_env_at_in(head_point_t,
                                    padded_full_envs,
                                    full_env_origin,
                                    padding,
                                    local_h_rows,
                                    local_w_cols,
                                    res)
    return local_env, local_env_origin
