#!/usr/bin/env python
import numpy as np

import ros_numpy
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from arm_robots_msgs.srv import GrippersTrajectory, GrippersTrajectoryRequest, GrippersTrajectoryResponse
from peter_msgs.srv import GetDualGripperPoints, \
    GetDualGripperPointsRequest, GetDualGripperPointsResponse
from rosgraph.names import ns_join


def interpolate_dual_gripper_trajectory(step_size: float,
                                        get_response: GetDualGripperPointsResponse,
                                        start_end_trajectory_request: GrippersTrajectoryRequest):
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

    n_steps = max(np.int64(distance / step_size), 5)
    waypoints = []
    gripper_waypoints = np.linspace(current_gripper_point, target_gripper_point, n_steps)

    return zip(waypoints)


class RosTfTrajectoryNode:
    def __init__(self):
        rospy.init_node('ros_tf_trajectory_node')
        self.in_srv = rospy.Service('ros_tf_trajectory', RosTfTrajectory, self.in_srv_cb)
        self.get_srv = rospy.ServiceProxy(, ?)
        self.tf2_wrapper = TF2Wrapper()
        self.step_size = 0.002

        rospy.spin()

    def in_srv_cb(self, req: GrippersTrajectoryRequest):
        settling_time_seconds = 0.02
        get_req = GetDualGripperPointsRequest()
        get_res = self.get_srv(get_req)
        waypoints = interpolate_dual_gripper_trajectory(step_size=self.step_size,
                                                        get_response=get_res,
                                                        start_end_trajectory_request=req)
        for waypoint in waypoints:
            for frame, translation in foobar.items():
                self.tf2_wrapper.send_transform(translation,
                                                [0, 0, 0, 1],
                                                parent='world',
                                                child=frame)
            rospy.sleep(settling_time_seconds)

        return GrippersTrajectoryResponse()


if __name__ == '__main__':
    forwarder = RosTfTrajectoryNode()
