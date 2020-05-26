from unittest import TestCase

from link_bot_pycommon.link_bot_sdf_utils import inflate_tf, OccupancyData


class Test(TestCase):
    def test_inflate_tf(self):
        # TODO
        env = OccupancyData(data=in_data)
        inflate_tf(env, radius_m=0.1)
