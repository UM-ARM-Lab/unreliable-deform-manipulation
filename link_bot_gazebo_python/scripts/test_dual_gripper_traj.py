import rospy
from geometry_msgs.msg import Point
import argparse
import numpy as np
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pause", action='store_true')

    args = parser.parse_args()

    rospy.init_node('test_dual_gripper_action')
    s = rospy.ServiceProxy('execute_dual_gripper_action', DualGripperTrajectory)

    extent = np.array([
        [-0.8, 0.8],
        [0.1, 1.6],
        [-0.2, 1.]
    ])

    np.random.seed(0)

    i = 0
    while True:
        i += 1
        p1 = np.random.uniform(extent[:, 0], extent[:, 1])
        p2 = np.random.uniform(extent[:, 0], extent[:, 1])

        req = DualGripperTrajectoryRequest()
        point1 = Point()
        point1.x = p1[0]
        point1.y = p1[1]
        point1.z = p1[2]
        point2 = Point()
        point2.x = p2[0]
        point2.y = p2[1]
        point2.z = p2[2]
        req.gripper1_points.append(point1)
        req.gripper2_points.append(point2)

        print(i)
        # if i < 120:
        #     continue

        s(req)
        if args.pause:
            key = input("press enter for new sample")
            if key == 'q':
                break


if __name__ == "__main__":
    main()
