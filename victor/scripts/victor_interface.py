#! /usr/bin/env python

import numpy as np

from ros_numpy import numpify
import tf2_ros
from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from peter_msgs.srv import GetDualGripperPoints, GetDualGripperPointsRequest, GetDualGripperPointsResponse, GetRopeState, GetRopeStateResponse, GetRopeStateRequest
import rospy


class CDCPDGetStateNode:

    def __init__(self):
        rospy.init_node("cdcpd_get_state")
        self.get_grippers_srv = rospy.Service(
            "get_dual_gripper_points", GetDualGripperPoints, self.get_dual_gripper_points_callback)
        self.get_rope_srv = rospy.Service("get_rope_state", GetRopeState, self.get_rope_state_callback)

        self.cdcpd_sub = rospy.Subscriber("cdcpd/output", PointCloud2, self.cdcpd_callback)
        self.latest_cdcpd_output = None

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

    def cdcpd_callback(self, output: PointCloud2):
        self.latest_cdcpd_output = output

    def get_rope_state_callback(self, req: GetRopeStateRequest):
        if self.latest_cdcpd_output is None:
            rospy.logwarn("No CDPCD output available")
            raise rospy.ServiceException("No CDPCD output available")
        cdcpd_points = numpify(self.latest_cdcpd_output)
        res = GetRopeStateResponse()
        for cdcpd_point in cdcpd_points:
            point = Point()
            point.x = cdcpd_point[0]
            point.y = cdcpd_point[1]
            point.z = cdcpd_point[2]
            res.positions.append(point)
        return res

    def get_dual_gripper_points_callback(self, req: GetDualGripperPointsRequest):
        res = GetDualGripperPointsResponse()

        # lookup TF of left and right gripper tool frames
        left_gripper_transform = self.buffer.lookup_transform(
            "world_origin", 'left_gripper_tool', rospy.Time.now, rospy.Duration(1))
        res.gripper1.x = left_gripper_transform.transform.translation.x
        res.gripper1.y = left_gripper_transform.transform.translation.y
        res.gripper1.z = left_gripper_transform.transform.translation.z

        right_gripper_transform = self.buffer.lookup_transform(
            "world_origin", "right_gripper_tool", rospy.Time.now, rospy.Duration(1))
        res.gripper2.x = right_gripper_transform.transform.translation.x
        res.gripper2.y = right_gripper_transform.transform.translation.y
        res.gripper2.z = right_gripper_transform.transform.translation.z
        return res


if __name__ == "__main__":
    n = CDCPDGetStateNode()

    # TESTING
    scenario = DualFloatingGripperRopeScenario()
    while True:
        state = scenario.get_state()
        scenario.plot_state_rviz(state, label="observed")
        rospy.sleep(0.1)
