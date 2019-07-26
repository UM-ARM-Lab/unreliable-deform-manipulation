#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from link_bot_gazebo.srv import LinkBotStateRequest, LinkBotState, LinkBotStateResponse
from time import sleep
get_state = rospy.ServiceProxy("/link_bot_state", LinkBotState)
repub = rospy.Publisher("/link_bot_image", Image, queue_size=10, latch=True)
rospy.init_node("testing")
req = LinkBotStateRequest()
while True:
    response = get_state.call(req)
    repub.publish(response.camera_image)
    sleep(0.1)
    
