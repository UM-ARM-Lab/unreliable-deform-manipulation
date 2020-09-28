#!/usr/bin/env python
import numpy as np

import ros_numpy
import rospy
from arm_robots_msgs.srv import GrippersTrajectory, GrippersTrajectoryRequest, GrippersTrajectoryResponse
from geometry_msgs.msg import Point
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, GetDualGripperPointsRequest


def interpolate_dual_gripper_trajectory(step_size: float, get_response, start_end_trajectory_request: GrippersTrajectoryRequest):
    """
    arg: step_size meters
    """
    current_gripper1_point = ros_numpy.numpify(get_response.gripper1)
    target_gripper1_point = ros_numpy.numpify(start_end_trajectory_request.grippers[0].points[0])

    current_gripper2_point = ros_numpy.numpify(get_response.gripper2)
    target_gripper2_point = ros_numpy.numpify(start_end_trajectory_request.grippers[1].points[0])

    gripper1_displacement = current_gripper1_point - target_gripper1_point
    gripper2_displacement = current_gripper2_point - target_gripper2_point

    distance = max(np.linalg.norm(gripper1_displacement), np.linalg.norm(gripper2_displacement))

    waypoint_traj_req = DualGripperTrajectoryRequest()
    n_steps = max(np.int64(distance / step_size), 5)
    settling_time_seconds = step_size / start_end_trajectory_request.speed
    waypoint_traj_req.settling_time_seconds = settling_time_seconds
    for gripper1_waypoint in np.linspace(current_gripper1_point, target_gripper1_point, n_steps):
        waypoint_traj_req.gripper1_points.append(ros_numpy.msgify(Point, gripper1_waypoint))
    for gripper2_waypoint in np.linspace(current_gripper2_point, target_gripper2_point, n_steps):
        waypoint_traj_req.gripper2_points.append(ros_numpy.msgify(Point, gripper2_waypoint))
    return waypoint_traj_req


class DualGripperActionForwarder:
    def __init__(self):
        rospy.init_node('test_traj_srv')
        self.out_srv = rospy.ServiceProxy('execute_dual_gripper_trajectory', DualGripperTrajectory)
        self.in_srv = rospy.Service('execute_dual_gripper_action', GrippersTrajectory, self.in_srv_cb)
        self.get_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)
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
