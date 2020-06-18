import numpy as np

import rospy
from peter_msgs.srv import JointTraj, JointTrajRequest
from trajectory_msgs.msg import JointTrajectoryPoint

rospy.init_node('test_traj_srv')
s = rospy.ServiceProxy('joint_traj', JointTraj)

req = JointTrajRequest()
req.traj.joint_names = ['victor::victor_right_arm_joint_4']
for angle in np.linspace(0, np.pi, 1000):
    point = JointTrajectoryPoint()
    point.positions = [angle]
    req.traj.points.append(point)
req.settling_time_seconds = 0.01
s(req)
