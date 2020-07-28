#!/usr/bin/env python
import argparse
import rospy
from gazebo_msgs.msg import LinkStates
from peter_msgs.srv import WorldControl, WorldControlRequest, GetJointState, GetJointStateRequest
from std_srvs.srv import Empty, EmptyRequest
from gazebo_msgs.srv import GetLinkState, GetLinkStateRequest, SetLinkState, SetLinkStateRequest
import rosbag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile")

    args = parser.parse_args()

    print("wating for service")
    world_control_srv = rospy.ServiceProxy("world_control", WorldControl)
    set_srv = rospy.ServiceProxy("gazebo/set_link_state", SetLinkState)
    joints_srv = rospy.ServiceProxy("joint_states", GetJointState)
    config_home_srv = rospy.ServiceProxy("configure_home", Empty)
    set_srv.wait_for_service()

    # run a few times to really make sure it happens
    for i in range(4):
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

    print("setting home position")

    # get the joint states and set the home position on the parameter server
    joints_res = joints_srv(GetJointStateRequest())
    left_arm_home = joints_res.joint_state.position[2:2+7]
    rospy.set_param("left_arm_home", left_arm_home)
    right_arm_home = joints_res.joint_state.position[2:2+7]
    rospy.set_param("right_arm_home", right_arm_home)
    torso_home = joints_res.joint_state.position[0:2]
    rospy.set_param("torso_home", torso_home)
    config_home_srv(EmptyRequest())

    print("done")


if __name__ == "__main__":
    main()
