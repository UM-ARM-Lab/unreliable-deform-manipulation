#!/usr/bin/env python
import numpy as np
import ros_numpy

import rospy
from geometry_msgs.msg import Point
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, \
    DualGripperTrajectoryResponse, GetDualGripperPointsRequest


def interpolate_dual_gripper_trajectory(step_size, get_response, start_end_trajectory_request):
    current_gripper1_point = ros_numpy.numpify(get_response.gripper1)
    target_gripper1_point = ros_numpy.numpify(start_end_trajectory_request.gripper1_points[0])

    current_gripper2_point = ros_numpy.numpify(get_response.gripper2)
    target_gripper2_point = ros_numpy.numpify(start_end_trajectory_request.gripper2_points[0])

    waypoint_traj_req = DualGripperTrajectoryRequest()
    waypoint_traj_req.settling_time_seconds = start_end_trajectory_request.settling_time_seconds
    gripper1_displacement = current_gripper1_point - target_gripper1_point
    gripper2_displacement = current_gripper2_point - target_gripper2_point
    distance = max(np.linalg.norm(gripper1_displacement), np.linalg.norm(gripper2_displacement))
    n_steps = max(np.int64(distance / step_size), 5)
    for gripper1_waypoint in np.linspace(current_gripper1_point, target_gripper1_point, n_steps):
        waypoint_traj_req.gripper1_points.append(ros_numpy.msgify(Point, gripper1_waypoint))
    for gripper2_waypoint in np.linspace(current_gripper2_point, target_gripper2_point, n_steps):
        waypoint_traj_req.gripper2_points.append(ros_numpy.msgify(Point, gripper2_waypoint))
    return waypoint_traj_req


class DualGripperActionForwarder:
    def __init__(self):
        rospy.init_node('test_traj_srv')
        self.out_srv = rospy.ServiceProxy('execute_dual_gripper_trajectory', DualGripperTrajectory)
        self.in_srv = rospy.Service('execute_dual_gripper_action', DualGripperTrajectory, self.in_srv_cb)
        self.get_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)
        self.step_size = 0.005

        rospy.spin()

    def in_srv_cb(self, req):
        get_req = GetDualGripperPointsRequest()
        get_res = self.get_srv(get_req)
        waypoint_traj_req = interpolate_dual_gripper_trajectory(step_size=self.step_size,
                                                                get_response=get_res,
                                                                start_end_trajectory_request=req)
        self.out_srv(waypoint_traj_req)
        return DualGripperTrajectoryResponse()
