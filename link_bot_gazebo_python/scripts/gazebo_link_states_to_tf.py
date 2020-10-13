#!/usr/bin/env python
import pathlib

import colorama
import hjson

import ros_numpy
import rospy
import tf
from gazebo_msgs.msg import LinkStates


class GazeboLinkStatesToTF:

    def __init__(self):
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.excluded_names = []
        if rospy.has_param('~config_file'):
            config_file = pathlib.Path(rospy.get_param("~config_file"))
            with config_file.open('r') as config_file:
                config = hjson.load(config_file)
            self.excluded_names = config['exclude']

        self.link_states_sub = rospy.Subscriber('gazebo/link_states', LinkStates, self.on_link_states)

    def on_link_states(self, msg: LinkStates):
        for link_name, link_pose in zip(msg.name, msg.pose):
            translation = ros_numpy.numpify(link_pose.position)
            rotation = ros_numpy.numpify(link_pose.orientation)
            child = "world"
            parent = link_name.replace("::", "/")
            if link_name not in self.excluded_names:
                self.tf_broadcaster.sendTransform(translation, rotation, rospy.Time.now(), parent, child)


def main():
    colorama.init(autoreset=True)
    rospy.init_node("gazebo_link_states_to_tf")

    GazeboLinkStatesToTF()

    rospy.spin()


if __name__ == '__main__':
    main()
