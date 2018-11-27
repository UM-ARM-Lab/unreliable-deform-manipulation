#!/usr/bin/env python
from __future__ import division, print_function

import rospy
from time import sleep
import numpy as np
import argparse
from gazebo_msgs.srv import GetLinkStateRequest, GetLinkState
from std_srvs.srv import Empty, EmptyRequest


class LogGazeboState:

    def __init__(self, args):
        rospy.init_node('LogWorldState')
        self.outfile = args.outfile
        self.model_name = args.model_name
        self.duration = args.duration
        self.verbose = args.verbose

    def run(self):
        DT = 0.1  # seconds per time step

        links = ['link_0', 'link_1', 'link_2', 'link_3', 'link_4', 'head']

        get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause(EmptyRequest())

        n_time_steps = int(self.duration / DT)
        n_links = len(links)
        data = np.ndarray((n_time_steps, 4*n_links))
        for i in range(n_time_steps):

            datum_indx = 0
            for link in links:
                link_state_req = GetLinkStateRequest()
                link_state_req.link_name = self.model_name + "::" + link
                link_state_resp = get_link_state(link_state_req)
                link_state = link_state_resp.link_state
                x = link_state.pose.position.x
                y = link_state.pose.position.y
                vx = link_state.twist.linear.x
                vy = link_state.twist.linear.y
                data[i, datum_indx] = x
                datum_indx += 1
                data[i, datum_indx] = y
                datum_indx += 1
                data[i, datum_indx] = vx
                datum_indx += 1
                data[i, datum_indx] = vy
                datum_indx += 1

            if self.verbose:
                print(data[i])

            sleep(DT)

        np.savetxt(self.outfile, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help="output file name", type=str)
    parser.add_argument("duration", help="number of seconds to log for", type=float, default=10)
    parser.add_argument("model_name")
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    teleop = LogGazeboState(args)
    teleop.run()
