#!/usr/bin/env python
from time import sleep

import rospy

from link_bot_gazebo.msg import ModelsEnable, ModelsPoses, ModelPose

if __name__ == '__main__':
    rospy.init_node('setup_tether')
    enable_pub = rospy.Publisher('/tether_enable', ModelsEnable, queue_size=1)
    tether_action_pub = rospy.Publisher('/tether_action', ModelsPoses, queue_size=1)

    while tether_action_pub.get_num_connections() < 1:
        pass

    for i in range(5):
        enable = ModelsEnable()
        enable.enable = True
        enable.model_names = ['link_bot']
        enable_pub.publish(enable)

        action = ModelPose()
        action.model_name = 'link_bot'
        action.pose.position.x = 0
        action.pose.position.y = 0

        actions = ModelsPoses()
        actions.actions = [action]
        tether_action_pub.publish(actions)

        sleep(0.1)
