import numpy as np
from colorama import Fore


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
            row, col = point_to_sdf_idx(px, py, sdf_data.resolution, sdf_data.origin)
            rope_images[i, row, col, j] = 1
    return rope_images
