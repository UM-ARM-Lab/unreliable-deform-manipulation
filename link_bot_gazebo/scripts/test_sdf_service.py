#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import timeit

import matplotlib.pyplot as plt
import rospy

from link_bot_gazebo import gazebo_services
from link_bot_pycommon import ros_pycommon

rospy.init_node("testing")

services = gazebo_services.GazeboServices()

local_env_data = ros_pycommon.get_local_occupancy_data(200, 200, 0.03, [0, 0], services)

plt.figure()
plt.imshow(local_env_data.image, extent=local_env_data.extent)
ax = plt.gca()
plt.show()
