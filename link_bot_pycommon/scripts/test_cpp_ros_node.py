#!/usr/bin/env python
from time import sleep

from colorama import Fore
from roscpp_initializer import init_node
import colorama
import rospy
from std_msgs.msg import Int64


# def cb(msg):
#     rospy.loginfo(Fore.GREEN + str(msg))


def main():
    colorama.init(autoreset=True)
    init_node("cpp_node_name")
    rospy.init_node("py_node_name")
    # sub = rospy.Subscriber("test_topic", Int64, cb)
    pub = rospy.Publisher("test_topic", Int64, queue_size=10)
    # calling this constructor should create the subscriber and we should start seeing messages from it
    i = 0
    while not rospy.is_shutdown():
        msg = Int64()
        msg.data = i
        rospy.loginfo(Fore.GREEN + f"Pub : {i}")
        pub.publish(msg)
        sleep(1)
        i = i + 1


if __name__ == '__main__':
    main()
