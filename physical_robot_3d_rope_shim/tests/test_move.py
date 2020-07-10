#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest

if __name__ == "__main__":
    rospy.init_node("test_move")
    rospy.wait_for_service("execute_dual_gripper_action")
    try:
        srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        req = DualGripperTrajectoryRequest()
        # Gazebo
        # req.gripper1_points.append(Point(1.13,  0.3, 0.95))
        # req.gripper2_points.append(Point(1.1, -0.4, 1.05))
        # req.gripper1_points.append(Point(1.1,  0.4, 1.05))
        # req.gripper2_points.append(Point(1.1, -0.4, 1.05))
        # req.gripper1_points.append(Point(1.1,  0.4, 1.05))
        # req.gripper2_points.append(Point(1.1, -0.4, 1.05))
        # ERB 1212
        req.gripper1_points.append(Point(2.0, -0.2, 0.9))
        req.gripper2_points.append(Point(2.0, -0.9, 1.0))
        req.gripper1_points.append(Point(2.0, -0.1, 1.0))
        req.gripper2_points.append(Point(2.0, -0.9, 1.0))
        req.gripper1_points.append(Point(2.0, -0.1, 1.0))
        req.gripper2_points.append(Point(2.0, -0.9, 1.0))
        resp = srv(req)
    except rospy.ServiceException as ex:
        print("Service call failed: %s" % ex)