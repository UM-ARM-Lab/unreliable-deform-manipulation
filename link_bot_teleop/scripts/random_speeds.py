#!/usr/bin/env python

import argparse
import numpy as np
from time import sleep

import rospy
from sensor_msgs.msg import Joy


class RandomSpeeds:

    def __init__(self, args):
        self.args = args
        rospy.init_node('random_speeds')

    def run(self):
        joy_pub = rospy.Publisher("/joy", Joy, queue_size=10)
        joy_msg = Joy()

        for i in range(self.args.num_seconds):
            joy_msg.axes = list(np.random.randn(2))
            joy_pub.publish(joy_msg)
            sleep(1)


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("num_seconds", type=int)

    args = parser.parse_args()

    teleop = RandomSpeeds(args)
    teleop.run()
