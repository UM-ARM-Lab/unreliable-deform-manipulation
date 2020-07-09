#!/usr/bin/env python

import random
import rospy
from geometry_msgs.msg import Point
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest

if __name__ == "__main__":
    random.seed(42)
    rospy.init_node("test_move")
    rospy.wait_for_service("execute_dual_gripper_action")
    try:
        srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        if True:
            req = DualGripperTrajectoryRequest()
            # Move the grippers to the same point, should stop short
            req.gripper1_points.append(Point(0.9, 0.0, 0.9))
            req.gripper2_points.append(Point(0.9, 0.0, 0.9))
            # Move the grippers down towards collision with objects on the table, should stop short
            req.gripper1_points.append(Point(1.0,  0.2, 0.8))
            req.gripper2_points.append(Point(1.0, -0.2, 0.8))
            # Move the grippers away from collision, should reach the targets
            req.gripper1_points.append(Point(0.9,  0.2, 1.05))
            req.gripper2_points.append(Point(0.9, -0.2, 1.05))
            # Move the grippers towards collision, should stop early
            req.gripper1_points.append(Point(0.5,  0.1, 0.8))
            req.gripper2_points.append(Point(0.5, -0.1, 0.8))
            # Move the grippers to freespace, should reach the targets
            req.gripper1_points.append(Point(0.7,  0.2, 0.8))
            req.gripper2_points.append(Point(0.7, -0.2, 0.8))
            resp = srv(req)

        # Generate random targets until Ctrl+C
        while not rospy.is_shutdown():
            req = DualGripperTrajectoryRequest()
            req.gripper1_points.append(Point(random.uniform(0.4, 1.2), random.uniform(-0.4, 0.4), random.uniform(0.6, 1.0)))
            req.gripper2_points.append(Point(random.uniform(0.4, 1.2), random.uniform(-0.4, 0.4), random.uniform(0.6, 1.0)))
            resp = srv(req)
    except rospy.ServiceException as ex:
        print("Service call failed: %s" % ex)
