#!/usr/bin/env python
import argparse
import pathlib

import rospy
from gazebo_msgs.srv import SetLinkState
from link_bot_gazebo_python.gazebo_services import GazeboServices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=pathlib.Path)

    args = parser.parse_args()

    to_delete = []
    bagfile_names = list(args.dir.iterdir())
    bagfile_names = sorted(bagfile_names)
    for bagfile_name in bagfile_names:
        if bagfile_name.suffix == '.bag':
            index = int(bagfile_name.stem[-4:])
            srv_name = "gazebo/set_link_state"
            rospy.loginfo(f"waiting for service {srv_name}")
            set_srv = rospy.ServiceProxy(srv_name, SetLinkState)
            set_srv.wait_for_service()

            gazebo_service_provider = GazeboServices()
            gazebo_service_provider.restore_from_bag(bagfile_name)

            r = input(f"{index:4d}: delete? [n/Y]")
            if r == 'Y' or r == 'y':
                to_delete.append(bagfile_name)

    print(to_delete)


if __name__ == '__main__':
    main()
