#!/usr/bin/env python
import rosservice
import argparse
import rospy
from gazebo_msgs.msg import LinkStates
from link_bot_gazebo_python.gazebo_services import GazeboServices
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

    print("resetting gazebo from bag file")

    gazebo_service_provider = GazeboServices()
    gazebo_service_provider.restore_from_bag(args.bagfile)

    print("setting home position")

    # get the joint states and set the home position on the parameter server
    joints_res = joints_srv(GetJointStateRequest())
    left_arm_home = joints_res.joint_state.position[2:2 + 7]
    rospy.set_param("left_arm_home", left_arm_home)
    right_arm_home = joints_res.joint_state.position[2:2 + 7]
    rospy.set_param("right_arm_home", right_arm_home)
    torso_home = joints_res.joint_state.position[0:2]
    rospy.set_param("torso_home", torso_home)
    if 'configure_home' in rosservice.get_service_list():
        config_home_srv(EmptyRequest())

    print("done")


if __name__ == "__main__":
    main()
