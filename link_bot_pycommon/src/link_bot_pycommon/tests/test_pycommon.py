from unittest import TestCase

from link_bot_pycommon.link_bot_pycommon import longest_reconverging_subsequence, trim_reconverging


class Test(TestCase):
    def test_start_and_end_of_max_consecutive_zeros(self):
        # contains no reconverging
        self.assertEqual(longest_reconverging_subsequence([]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([1]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([0]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([0, 0]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([1, 0]), (0, 0))
        self.assertEqual(longest_reconverging_subsequence([1, 0, 0]), (0, 0))
        # contains reconverging
        self.assertEqual(longest_reconverging_subsequence([0, 1, 0]), (0, 1))
        self.assertEqual(longest_reconverging_subsequence([0, 1, 0, 0]), (0, 1))
        self.assertEqual(longest_reconverging_subsequence([0, 0, 1]), (0, 2))
        self.assertEqual(longest_reconverging_subsequence([0, 0, 1, 0]), (0, 2))
        self.assertEqual(longest_reconverging_subsequence([1, 0, 0, 1]), (1, 3))
        self.assertEqual(longest_reconverging_subsequence([0, 0, 1, 0, 0, 0, 1, 0]), (3, 6))

    def test_trim_reconverging(self):
        self.assertEqual(trim_reconverging([1, 0, 1]), (0, 3))
        self.assertEqual(trim_reconverging([1, 0, 0, 1]), (0, 4))
        self.assertEqual(trim_reconverging([1, 0, 0, 1, 1]), (0, 5))
        self.assertEqual(trim_reconverging([1, 0, 0, 1]), (0, 4))
        self.assertEqual(trim_reconverging([1, 0, 0, 1, 1, 0]), (0, 5))
        self.assertEqual(trim_reconverging([1, 0, 1, 0, 0, 1, 1]), (2, 7))
        self.assertEqual(trim_reconverging([1, 0, 0, 0, 1]), (0, 5))
        self.assertEqual(trim_reconverging([1, 0, 1, 0, 1]), (0, 3))  # tie break goes to the first occurrence
        self.assertEqual(trim_reconverging([1, 0, 1, 0, 0, 1]), (2, 6))
        self.assertEqual(trim_reconverging([1, 1, 0, 0, 1, 1]), (0, 6))
        self.assertEqual(trim_reconverging([1, 0, 0, 1, 1, 0, 0, 0, 1, 0]), (3, 9))
        #                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
