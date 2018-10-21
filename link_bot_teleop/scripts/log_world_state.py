#!/usr/bin/env python

import rospy
from time import sleep
import numpy as np
import argparse
from deformable_manipulation_msgs.msg import WorldState

class LogBulletWorldState:

    def __init__(self, args):
        rospy.init_node('LogWorldState')
        rospy.Subscriber("/world_state", WorldState, self.callback)
        self.log = []
        self.outfile = args.outfile

    def spin(self):
        rospy.spin()

    def callback(self, data):
        d = np.array([[c.x, c.y] for c in data.object_configuration])
        d = d.flatten()
        self.log.append(d)

    def write(self):
        log_arr = np.array(self.log)
        np.savetxt(self.outfile, log_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help="output file name", type=str)
    parser.add_argument("--duration", '-d', help="number of seconds to log for", type=float, default=10)

    args = parser.parse_args()

    teleop = LogBulletWorldState(args)
    sleep(args.duration)
    teleop.write()
