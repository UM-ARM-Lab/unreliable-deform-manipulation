#!/usr/bin/env python
from time import sleep

import rospy
from geometry_msgs.msg import Point
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest

if __name__ == '__main__':
    rospy.init_node('start_config')
    srv = rospy.ServiceProxy('execute_dual_gripper_action', DualGripperTrajectory)
    rospy.wait_for_service('execute_dual_gripper_action')

    gripper1_point = Point()
    gripper1_point.x = 0.0
    gripper1_point.y = 0.0
    gripper1_point.z = 0.1

    gripper2_point = Point()
    gripper2_point.x = 0.2
    gripper2_point.y = 0.0
    gripper2_point.z = 0.1

    req = DualGripperTrajectoryRequest()
    req.gripper1_points.append(gripper1_point)
    req.gripper2_points.append(gripper2_point)
    req.settling_time_seconds = 0.5
    srv(req)

