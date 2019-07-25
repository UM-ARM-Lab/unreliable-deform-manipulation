#!/usr/bin/env python
from time import sleep

import rospy
import tf

if __name__ == '__main__':
    rospy.init_node('hog_tf_broadcaster')
    br = tf.TransformBroadcaster()

    while True:
        br.sendTransform((0, 0, 0),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         'hog_desired',
                         "world")
