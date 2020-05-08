from unittest import TestCase

from link_bot_pycommon.link_bot_pycommon import compute_max_consecutive_zeros


class Test(TestCase):
    def test_compute_max_consecutive_zeros(self):
        self.assertEqual(compute_max_consecutive_zeros([]), 0)
        self.assertEqual(compute_max_consecutive_zeros([1]), 0)
        self.assertEqual(compute_max_consecutive_zeros([0]), 1)
        self.assertEqual(compute_max_consecutive_zeros([1, 0]), 1)
        self.assertEqual(compute_max_consecutive_zeros([0, 1, 0]), 1)
        self.assertEqual(compute_max_consecutive_zeros([0, 0]), 2)
        self.assertEqual(compute_max_consecutive_zeros([1, 0, 0]), 2)
        self.assertEqual(compute_max_consecutive_zeros([0, 0, 1]), 2)
        self.assertEqual(compute_max_consecutive_zeros([0, 0, 1, 0]), 2)
        self.assertEqual(compute_max_consecutive_zeros([0, 1, 0, 0]), 2)
        self.assertEqual(compute_max_consecutive_zeros([0, 0, 0]), 3)
        self.assertEqual(compute_max_consecutive_zeros([0, 0, 0, 1]), 3)
        self.assertEqual(compute_max_consecutive_zeros([0, 0, 1, 0, 0, 0, 1, 0]), 3)
