#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import timeit

import matplotlib.pyplot as plt
import numpy as np
import rospy

from link_bot_gazebo import gazebo_services
from link_bot_pycommon.ros_pycommon import get_local_occupancy_data

rospy.init_node("testing")

services = gazebo_services.GazeboServices()

h_rows = 50
w_cols = 50
res = 0.03

occupancy_data = get_local_occupancy_data(rows=h_rows,
                                          cols=w_cols,
                                          res=res,
                                          center_point=np.array([0.0, 0]), services=services)

plt.figure()
print(occupancy_data.extent)
plt.imshow(occupancy_data.image, extent=occupancy_data.extent)
ax = plt.gca()
ax.set_xticks(np.arange(occupancy_data.extent[0], occupancy_data.extent[1], res))
ax.set_yticks(np.arange(occupancy_data.extent[2], occupancy_data.extent[3], res))
ax.set_xticklabels([])
ax.set_yticklabels([])
# plt.grid()
plt.show()
