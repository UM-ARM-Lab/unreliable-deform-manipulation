#!/usr/bin/env python
import moveit_commander
from colorama import Fore
import colorama
import pycpp_ros_sub
import rospy
# from std_msgs.msg import Int64


# def cb(msg):
#     rospy.loginfo(Fore.GREEN + str(msg))


def main():
    colorama.init(autoreset=True)
    moveit_commander.roscpp_initialize([])
    # rospy.init_node("test_node")
    pycpp_sub = pycpp_ros_sub.PyCppRosSub()
    print("shutting down...")
    moveit_commander.roscpp_shutdown()
    # sub = rospy.Subscriber("test_topic", Int64, cb)
    # pub = rospy.Publisher("test_topic", Int64, queue_size=10)
    # calling this constructor should create the subscriber and we should start seeing messages from it
    # i = 0
    # while not rospy.is_shutdown():
    #     msg = Int64
    #     msg.data = i
    #     rospy.loginfo(Fore.GREEN + f"Pub : {i}")
    #     pub.publish(msg)
    #     rospy.sleep(1)
    #     i = i + 1


if __name__ == '__main__':
    main()
