#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import timeit

import matplotlib.pyplot as plt
import numpy as np
import rospy

from link_bot_gazebo import gazebo_utils
# noinspection PyUnresolvedReferences
from link_bot_gazebo.gazebo_utils import get_sdf_data, get_local_sdf_data

rospy.init_node("testing")

services = gazebo_utils.GazeboServices()

# print(timeit.timeit('get_sdf_data(3, 3, 0.03, services)', number=1000, globals=globals()))

h_rows = 100
w_cols = 100
res = 0.03
sdf_data = get_local_sdf_data(sdf_rows=h_rows, sdf_cols=w_cols, res=res, origin_point=np.array([0, -1]), services=services)

plt.figure()
plt.imshow(sdf_data.image < 0.0, extent=sdf_data.extent)
ax = plt.gca()
# ax.set_xticks(np.arange(sdf_data.extent[0], sdf_data.extent[1], res))
# ax.set_yticks(np.arange(sdf_data.extent[2], sdf_data.extent[3], res))
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# plt.grid()
plt.show()
