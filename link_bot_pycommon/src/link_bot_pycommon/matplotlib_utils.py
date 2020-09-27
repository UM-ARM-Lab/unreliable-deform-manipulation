import numpy as np
from matplotlib import cm


def save_unconstrained_layout(fig, filename, dpi=300):
    fig.set_constrained_layout(False)
    fig.savefig(filename, bbox_inches='tight', dpi=100)


def state_image_to_cmap(state_image: np.ndarray, cmap=cm.viridis, binary_threshold=0.1):
    h, w, n_channels = state_image.shape
    new_image = np.zeros([h, w, 3])
    for channel_idx in range(n_channels):
        channel = np.take(state_image, indices=channel_idx, axis=-1)
        color = cmap(channel_idx / n_channels)[:3]
        rows, cols = np.where(channel > binary_threshold)
        new_image[rows, cols] = color
    return new_image


def paste_over(i1, i2, binary_threshold=0.1):
    # first create a mask for everywhere i1 > binary_threshold, and zero out those pixels in i2, then add.
    mask = np.any(i1 > binary_threshold, axis=2)
    i2[mask] = 0
    return i2 + i1
