#!/usr/bin/env python
import numpy as np

import ros_numpy
import rospy
from arm_robots_msgs.srv import GrippersTrajectory, GrippersTrajectoryRequest, GrippersTrajectoryResponse
from geometry_msgs.msg import Point
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, GetDualGripperPointsRequest
from rosgraph.names import ns_join


def interpolate_dual_gripper_trajectory(step_size: float, get_response, start_end_trajectory_request: GrippersTrajectoryRequest):
    """
    arg: step_size meters
    """
    current_left_gripper_point = ros_numpy.numpify(get_response.left_gripper)
    target_left_gripper_point = ros_numpy.numpify(start_end_trajectory_request.grippers[0].points[0])

    current_right_gripper_point = ros_numpy.numpify(get_response.right_gripper)
    target_right_gripper_point = ros_numpy.numpify(start_end_trajectory_request.grippers[1].points[0])

    left_gripper_displacement = current_left_gripper_point - target_left_gripper_point
    right_gripper_displacement = current_right_gripper_point - target_right_gripper_point

    distance = max(np.linalg.norm(left_gripper_displacement), np.linalg.norm(right_gripper_displacement))

    waypoint_traj_req = DualGripperTrajectoryRequest()
    n_steps = max(np.int64(distance / step_size), 5)
    waypoint_traj_req.settling_time_seconds = 0.01
    for left_gripper_waypoint in np.linspace(current_left_gripper_point, target_left_gripper_point, n_steps):
        waypoint_traj_req.left_gripper_points.append(ros_numpy.msgify(Point, left_gripper_waypoint))
    for right_gripper_waypoint in np.linspace(current_right_gripper_point, target_right_gripper_point, n_steps):
        waypoint_traj_req.right_gripper_points.append(ros_numpy.msgify(Point, right_gripper_waypoint))
    return waypoint_traj_req


class DualGripperActionForwarder:
    def __init__(self):
        rospy.init_node('test_traj_srv')
        rope_ns = rospy.get_param("~rope_ns")
        self.out_srv = rospy.ServiceProxy(ns_join(rope_ns, 'execute_dual_gripper_trajectory'), DualGripperTrajectory)
        self.in_srv = rospy.Service(ns_join(rope_ns, 'execute_dual_gripper_action'), GrippersTrajectory, self.in_srv_cb)
        self.get_srv = rospy.ServiceProxy(ns_join(rope_ns, "get_dual_gripper_points"), GetDualGripperPoints)
        self.step_size = 0.002

        rospy.spin()

    def in_srv_cb(self, req: GrippersTrajectoryRequest):
        get_req = GetDualGripperPointsRequest()
        get_res = self.get_srv(get_req)
        waypoint_traj_req = interpolate_dual_gripper_trajectory(step_size=self.step_size,
                                                                get_response=get_res,
                                                                start_end_trajectory_request=req)
        self.out_srv(waypoint_traj_req)
        return GrippersTrajectoryResponse()
