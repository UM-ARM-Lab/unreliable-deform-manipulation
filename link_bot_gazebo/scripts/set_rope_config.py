#!/usr/bin/env python
import argparse
import numpy as np
from link_bot_pycommon.args import my_formatter
from link_bot_gazebo.gazebo_services import GazeboServices
import rospy


def main():
    rospy.init_node("set_rope_config")
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("joint_angles", nargs='+', type=float, help='in degrees')
    parser.add_argument("-x", type=float, default=0, help='x')
    parser.add_argument("-y", type=float, default=0, help='y')
    parser.add_argument("--yaw", "-Y", type=float, default=0, help='yaw')

    args = parser.parse_args()

    joint_angles_rad = np.deg2rad(args.joint_angles)
    service_provider = GazeboServices([])
    service_provider.reset_rope(x=args.x,
                                y=args.y,
                                yaw=args.yaw,
                                joint_angles=joint_angles_rad)


if __name__ == '__main__':
    main()
