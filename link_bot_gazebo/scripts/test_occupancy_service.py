#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import timeit

import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_gazebo import gazebo_services
from link_bot_pycommon.ros_pycommon import get_local_occupancy_data

rospy.init_node("testing")

services = gazebo_services.GazeboServices([])

h_rows = 100
w_cols = 100
res = 0.01

while True:
    occupancy_data = get_local_occupancy_data(rows=h_rows,
                                              cols=w_cols,
                                              res=res,
                                              center_point=np.array([0.6, 0.6]),
                                              service_provider=services,
                                              robot_name='car')

    plt.figure()
    plt.imshow(occupancy_data.image, extent=occupancy_data.extent)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
