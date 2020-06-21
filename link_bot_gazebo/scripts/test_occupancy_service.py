#!/usr/bin/env python
from time import sleep

import rospy
from link_bot_gazebo import gazebo_services
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from mps_shape_completion_msgs.msg import OccupancyStamped

rospy.init_node("test_occupancy_service")
pub = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)

services = gazebo_services.GazeboServices()

res = 0.03
extent_3d = [-0.3, 0.3, -0.3, 0.3, 0.03, 0.6]
while True:
    try:
        environment = get_environment_for_extents_3d(extent=extent_3d,
                                                     res=res,
                                                     service_provider=services,
                                                     robot_name='link_bot')
        print(link_bot_sdf_utils.idx_to_point_3d(0, 0, 0, res, environment['origin']))
        msg = link_bot_sdf_utils.environment_to_occupancy_msg(environment)
        print(msg.header.stamp)
        pub.publish(msg)
        sleep(1.0)
    except rospy.ServiceException:
        pass
