#!/usr/bin/env python2
import matplotlib.pyplot as plt
import numpy as np
import rospy

from link_bot_sdf_tools import link_bot_sdf_tools
from link_bot_sdf_tools.srv import ComputeSDF

rospy.init_node("testing")

s = rospy.ServiceProxy("/sdf", ComputeSDF)

sdf_data = link_bot_sdf_tools.request_sdf_data(s, width=2, height=2, res=0.01)

plt.figure()
plt.imshow(sdf_data.image < 0.0, extent=sdf_data.extent)

plt.show()
