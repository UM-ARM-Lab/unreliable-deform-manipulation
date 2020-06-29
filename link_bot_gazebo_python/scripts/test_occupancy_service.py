#!/usr/bin/env python
from time import sleep

import rospy
from link_bot_gazebo import gazebo_services
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from mps_shape_completion_msgs.msg import OccupancyStamped
import tf2_ros


rospy.init_node("test_occupancy_service")
pub = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)

services = gazebo_services.GazeboServices()
broadcaster = tf2_ros.StaticTransformBroadcaster()

res = 0.03

extent_3d = [0.4, 1.2, -0.72, 0.72, 0.5, 1.5]
# extent_3d = [-0.3, 0.3, -0.3, 0.3, 0.00, 0.3]
while True:
    try:
        environment = get_environment_for_extents_3d(extent=extent_3d,
                                                     res=res,
                                                     service_provider=services,
                                                     robot_name='victor_and_rope::link_bot')
        print(link_bot_sdf_utils.idx_to_point_3d(0, 0, 0, res, environment['origin']))
        msg = link_bot_sdf_utils.environment_to_occupancy_msg(environment)
        print(msg.header.stamp)

        link_bot_sdf_utils.send_occupancy_tf(broadcaster, environment)
        pub.publish(msg)

        sleep(1.0)
    except rospy.ServiceException:
        pass
