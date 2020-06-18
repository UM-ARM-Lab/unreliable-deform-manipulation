import rospy
from geometry_msgs.msg import Point
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest

rospy.init_node('test_traj_srv')
s = rospy.ServiceProxy('execute_dual_gripper_trajectory', DualGripperTrajectory)

req = DualGripperTrajectoryRequest()
req.settling_time_seconds = 0.5
x = 0
for i in range(15):
    point1 = Point()
    point1.x = x + 0.3 - i * 0.02
    point1.y = 0
    point1.z = 0.5
    point2 = Point()
    point2.x = x
    point2.y = 0
    point2.z = 0.5
    req.gripper1_points.append(point1)
    req.gripper2_points.append(point2)
    x += 0.01

s(req)
