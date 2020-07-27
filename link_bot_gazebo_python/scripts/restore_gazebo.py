#!/usr/bin/env python
import argparse
import rospy
from gazebo_msgs.msg import LinkStates
from peter_msgs.srv import WorldControl, WorldControlRequest
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest, SetLinkState, SetLinkStateRequest
import rosbag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile")

    args = parser.parse_args()

    print("wating for service")
    world_control_srv = rospy.ServiceProxy("world_control", WorldControl)
    set_srv = rospy.ServiceProxy("gazebo/set_link_state", SetLinkState)
    set_srv.wait_for_service()

    bag = rosbag.Bag(args.bagfile)
    _, msg, _ = next(bag.read_messages())
    bag.close()

    print("read bagfile")

    n = len(msg.name)
    for i in range(n):
        name = msg.name[i]
        pose = msg.pose[i]
        twist = msg.twist[i]
        set_req = SetLinkStateRequest()
        set_req.link_state.link_name = name
        set_req.link_state.pose = pose
        set_req.link_state.twist = twist
        set_srv(set_req)

    step = WorldControlRequest()
    step.steps = 1
    world_control_srv(step)

    print("done")


if __name__ == "__main__":
    main()
