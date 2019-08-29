#!/usr/bin/env python
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import rospy
from sensor_msgs.msg import Image

from link_bot_gazebo.srv import LinkBotStateRequest, LinkBotState


def main():
    get_state = rospy.ServiceProxy("/link_bot_state", LinkBotState)
    repub = rospy.Publisher("/link_bot_image", Image, queue_size=10, latch=True)
    rospy.init_node("testing")
    req = LinkBotStateRequest()

    while True:
        try:
            response = get_state.call(req)
            repub.publish(response.camera_image)
            sleep(0.01)
        except rospy.service.ServiceException:
            pass


if __name__ == '__main__':
    main()
