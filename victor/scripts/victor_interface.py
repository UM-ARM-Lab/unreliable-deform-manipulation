#! /usr/bin/env python

import numpy as np

import rospy
from victor_hardware_interface.msg import ControlMode
from tf.transformations import compose_matrix

from arm_or_robots import motion_victor

config_start = [0.527, 1.378, 1.278, 1.153, -2.438, -0.64, 0.43]


def grab_cloth(mev):
    """Pickup and drop a cloth from the table"""
    # mev.set_gripper("right", [0.3, 0.3, 0.3], blocking=False)
    mev.set_manipulator("right_arm")
    mev.change_control_mode(ControlMode.JOINT_IMPEDANCE)

    # raw_input("give me the rope plz!")
    mev.set_gripper("right", [0.38, 0.36, 0.36], blocking=True)

    mev.plan_to_configuration(config_start, execute=True, blocking=True)
    # mev.guarded_move_hand_straight([-1, 0, 0], 0.1, force_trigger=15, step_size=0.01)

    mev.guarded_move_hand_straight([0, 1, 0], 0.1, force_trigger=15, step_size=0.01)


def run_mev():
    rospy.init_node("motion")
    mev = motion_victor.MotionEnabledVictor()

    grab_cloth(mev)


if __name__ == "__main__":
    run_mev()
