#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
import rospy

from link_bot_pycommon import link_bot_pycommon
from sdf_tools.srv import ComputeSDF

rospy.init_node("testing")

s = rospy.ServiceProxy("/sdf", ComputeSDF)

sdf_data = link_bot_pycommon.request_sdf_data(s)

plt.figure()
plt.imshow(sdf_data.image, extent=sdf_data.extent)

plt.figure()
plt.imshow(np.flipud(sdf_data.sdf.T), extent=sdf_data.extent)
subsample = 2
x_range = np.arange(sdf_data.extent[0], sdf_data.extent[1], subsample * sdf_data.resolution[0])
y_range = np.arange(sdf_data.extent[0], sdf_data.extent[1], subsample * sdf_data.resolution[1])
y, x = np.meshgrid(y_range, x_range)
dx = sdf_data.gradient[::subsample, ::subsample, 0]
dy = sdf_data.gradient[::subsample, ::subsample, 1]
plt.quiver(x, y, dx, dy, units='x', scale=10)

plt.show()
