from unittest import TestCase

from geometry_msgs.msg import Point
from link_bot_gazebo.dual_gripper_action_forwarder_lib import interpolate_dual_gripper_trajectory
from peter_msgs.srv import DualGripperTrajectoryRequest


def make_point(x, y, z):
    p = Point()
    p.x = x
    p.y = y
    p.z = z
    return p


class TestDualGripperActionForwarder(TestCase):
    def test_interpolate_dual_gripper_trajectory(self):
        in_gripper1_start = make_point(0.3, 0, 0)
        in_gripper1_end = make_point(0.4, 0, 0)

        in_gripper2_start = make_point(0.0, 0, 0)
        in_gripper2_end = make_point(0.0, 0.1, -0.1)

        in_req = DualGripperTrajectoryRequest()
        in_req.settling_time_seconds = 1.0

        in_req.gripper1_points.append(in_gripper1_start)
        in_req.gripper1_points.append(in_gripper1_end)

        in_req.gripper2_points.append(in_gripper2_start)
        in_req.gripper2_points.append(in_gripper2_end)

        out_req = interpolate_dual_gripper_trajectory(step_size=0.01, start_end_trajectory_request=in_req)

        self.assertEqual(len(out_req.gripper1_points), 10)
        self.assertEqual(len(out_req.gripper2_points), 10)
        self.assertEqual(out_req.settling_time_seconds, 1.0)
        self.assertEqual(out_req.gripper1_points[0], in_gripper1_start)
        self.assertEqual(out_req.gripper1_points[-1], in_gripper1_end)
        self.assertEqual(out_req.gripper2_points[0], in_gripper2_start)
        self.assertEqual(out_req.gripper2_points[-1], in_gripper2_end)
