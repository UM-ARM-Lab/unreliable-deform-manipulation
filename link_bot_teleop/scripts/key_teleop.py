#!/usr/bin/env python

import argparse
import numpy as np
from time import sleep

import rospy
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest, GetLinkState, GetLinkStateRequest


class LinkBotTeleop:

    def __init__(self, args):
        self.args = args
        rospy.init_node('LinkBotTeleop')
        self.model_name = args.model_name

    def run(self):
        DT = 0.1  # seconds per time step

        wrench_req = ApplyBodyWrenchRequest()
        wrench_req.body_name = self.model_name + "::link_1"
        wrench_req.reference_frame = "world"
        wrench_req.duration.secs = -1
        wrench_req.duration.nsecs = -1

        links = ['link_0', 'link_1']

        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)

        inputs = [[self.args.vx, self.args.vy]] * 20

        data = []
        for fx, fy in inputs:

            wrench_req.wrench.force.x = fx
            wrench_req.wrench.force.y = fy

            apply_wrench(wrench_req)

            datum = []
            for link in links:
                link_state_req = GetLinkStateRequest()
                link_state_req.link_name = link
                link_state_resp = get_link_state(link_state_req)
                link_state = link_state_resp.link_state
                x = link_state.pose.position.x
                y = link_state.pose.position.y
                datum.extend([x, y])

            link_state_req = GetLinkStateRequest()
            link_state_req.link_name = 'link_1'
            link_state_resp = get_link_state(link_state_req)
            link_state = link_state_resp.link_state
            vx = link_state.twist.linear.x
            vy = link_state.twist.linear.y
            datum.extend([vx, vy])

            datum = np.array(datum)

            if self.args.verbose:
                print(datum)
            data.append(datum)

            sleep(DT)

        wrench_req.wrench.force.x = 0
        wrench_req.wrench.force.y = 0

        apply_wrench(wrench_req)

        np.savetxt(self.args.outfile, data)


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help='filename to store data in')
    parser.add_argument("vx", type=float)
    parser.add_argument("vy", type=float)
    parser.add_argument("--model_name", '-m', default="link_bot")
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    teleop = LinkBotTeleop(args)
    teleop.run()
