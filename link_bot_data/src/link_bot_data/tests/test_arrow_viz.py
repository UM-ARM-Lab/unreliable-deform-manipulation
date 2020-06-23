from time import sleep

import numpy as np

import rospy
from link_bot_data.visualization import rviz_arrow
from visualization_msgs.msg import MarkerArray

if __name__ == '__main__':
    rospy.init_node('test_arrow_viz')
    action_viz_pub = rospy.Publisher("action_viz", MarkerArray, queue_size=10)
    r = 1
    g = 0
    b = 0
    a = 1
    s1 = np.array([0.5, 0.0, 0.0])

    a1 = np.array([0.0, 0.0, 0.0])
    msg = MarkerArray()
    arrow = rviz_arrow(s1, a1, r, g, b, 1)
    msg.markers.append(arrow)
    print(msg)

    for i in range(5):
        action_viz_pub.publish(msg)
        sleep(0.5)
