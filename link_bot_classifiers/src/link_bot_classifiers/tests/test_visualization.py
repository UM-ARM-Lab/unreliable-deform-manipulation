from unittest import TestCase

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from link_bot_classifiers.visualization import state_image_to_cmap, paste_over


class Test(TestCase):
    def test_state_image_to_cmap(self):
        in_image = np.zeros([3, 3, 5], dtype=np.float32)
        in_image[0, 0, 0] = 1
        in_image[0, 1, 1] = 1
        in_image[0, 2, 2] = 1
        in_image[1, 2, 3] = 1
        in_image[2, 2, 4] = 1
        cmap = cm.Greys

        out_image = state_image_to_cmap(in_image, cmap=cmap)

        expected_image = np.zeros([3, 3, 3], dtype=np.float32)
        expected_image[0, 0] = cmap(0 / 5)[:3]
        expected_image[0, 1] = cmap(1 / 5)[:3]
        expected_image[0, 2] = cmap(2 / 5)[:3]
        expected_image[1, 2] = cmap(3 / 5)[:3]
        expected_image[2, 2] = cmap(4 / 5)[:3]
        np.testing.assert_allclose(out_image, expected_image)

    def test_state_image_to_cmap2(self):
        in_image = np.zeros([3, 3, 3], dtype=np.float32)
        in_image[0, 0, 0] = 1
        in_image[1, 1, 1] = 1
        in_image[2, 2, 2] = 1
        cmap = cm.viridis

        out_image = state_image_to_cmap(in_image, cmap=cmap)

        expected_image = np.zeros([3, 3, 3], dtype=np.float32)
        expected_image[0, 0] = cmap(0 / 3)[:3]
        expected_image[1, 1] = cmap(1 / 3)[:3]
        expected_image[2, 2] = cmap(2 / 3)[:3]
        np.testing.assert_allclose(out_image, expected_image)

    def test_paste_over(self):
        i1 = np.zeros([3, 3, 3], dtype=np.float32)
        i1[0, 1] = 1
        i1[1, 1] = 1
        i1[2, 1] = 1
        i2 = np.zeros([3, 3, 3], dtype=np.float32)
        i2[0, 2] = 1
        i2[1, 2] = 1
        i2[2, 2] = 1
        out_image = paste_over(i1, i2)

        expected_image = np.zeros([3, 3, 3], dtype=np.float32)
        expected_image[0, 1] = 1
        expected_image[1, 1] = 1
        expected_image[2, 1] = 1
        expected_image[0, 2] = 1
        expected_image[1, 2] = 1
        expected_image[2, 2] = 1
        np.testing.assert_allclose(out_image, expected_image)

    def test_paste_over(self):
        i1 = np.zeros([3, 3, 3], dtype=np.float32)
        i1[0, 1] = 1
        i1[1, 1] = 0.5
        i1[2, 1] = 0.5
        i1[2, 2] = 0.75
        i2 = np.zeros([3, 3, 3], dtype=np.float32)
        i2[0, 1] = 1
        i2[1, 1] = 1
        i2[2, 1] = 1
        i2[0, 0] = 0.75
        out_image = paste_over(i1, i2)

        expected_image = np.zeros([3, 3, 3], dtype=np.float32)
        expected_image[0, 1] = 1
        expected_image[1, 1] = 0.5
        expected_image[2, 1] = 0.5
        expected_image[0, 0] = 0.75
        expected_image[2, 2] = 0.75
        fig, axes = plt.subplots(2)
        axes[0].imshow(expected_image)
        axes[1].imshow(out_image)
        plt.show()
        np.testing.assert_allclose(out_image, expected_image)
