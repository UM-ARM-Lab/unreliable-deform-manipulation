#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import rospy

import rospy
from link_bot_gazebo import gazebo_services
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from mps_shape_completion_msgs.msg import OccupancyStamped

rospy.init_node("test_occupancy_service")
pub = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)

services = gazebo_services.GazeboServices([])

res = 0.03
while True:
    environment = get_environment_for_extents_3d([-1, 1, -1, 1, 0, 0.25],
                                                 res=res,
                                                 service_provider=services,
                                                 robot_name='link_bot')
    msg = environment_to_occupancy_msg(environment)
    print(msg.header)
    pub.publish(msg)
    rospy.sleep(1.0)
