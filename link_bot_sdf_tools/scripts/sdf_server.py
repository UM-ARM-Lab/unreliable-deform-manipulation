#!/usr/bin/env python
import rospy

from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_sdf_tools import link_bot_sdf_tools
from link_bot_sdf_tools.srv import ComputeSDF, ComputeSDF2Response, ComputeSDF2


class SDFServer:

    def __init__(self):
        rospy.init_node('sdf_server')
        self.services = GazeboServices()
        self.services.compute_sdf2 = rospy.Service('/sdf2', ComputeSDF2, self.compute_sdf2)
        self.services.compute_sdf = rospy.ServiceProxy('/sdf', ComputeSDF)
        self.services.wait(verbose=True)

        rospy.spin()

    def compute_sdf2(self, req):
        # req is empty
        del req
        w = 1
        h = 1
        resolution = 0.01
        sdf_data = link_bot_sdf_tools.request_sdf_data(self.services.compute_sdf, width=w, height=h, res=resolution)

        res = ComputeSDF2Response()
        res.w = 1
        res.h = 1
        res.origin = sdf_data.origin
        res.res = sdf_data.resolution
        res.sdf = sdf_data.sdf
        res.gradient = sdf_data.gradient
        return res


if __name__ == '__main__':
    SDFServer()
