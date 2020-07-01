#!/usr/bin/env python

# Mostly copied from kuka_iiwa_interface/victor_fake_hardware_interface/src/victor_fake_hardware_interface/minimal_fake_arm_interface.py

import rospy
from victor_hardware_interface_msgs.msg import *
from victor_hardware_interface_msgs.srv import *
from threading import Thread
from threading import Lock

arm_names = ["left_arm", "right_arm"]
cartesian_field_names = ["x", "y", "z", "a", "b", "c"]
joint_names = ['joint_' + str(i) for i in range(1, 8)]

control_mode_feedback_rate = 1.0 # Hz


def DefaultControlModeParametersStatus():
    """
    :return type: ControlModeParameters
    """
    msg = ControlModeParameters()

    msg.control_mode.mode = ControlMode.JOINT_POSITION

    # Joint impedance parameters
    for joint in joint_names:
        setattr(msg.joint_impedance_params.joint_stiffness, joint, 0.0)
        setattr(msg.joint_impedance_params.joint_damping, joint, 0.0)

    # Cartesian impedance parameters
    for field in cartesian_field_names:
        setattr(msg.cartesian_impedance_params.cartesian_stiffness, field, 0.1)
        setattr(msg.cartesian_impedance_params.cartesian_damping, field, 0.1)
    msg.cartesian_impedance_params.nullspace_stiffness = 0.0
    msg.cartesian_impedance_params.nullspace_damping = 0.3

    # Cartesian control mode limits
    for field in cartesian_field_names:
        setattr(msg.cartesian_control_mode_limits.max_path_deviation, field, 0.1)
        setattr(msg.cartesian_control_mode_limits.max_cartesian_velocity, field, 0.1)
        setattr(msg.cartesian_control_mode_limits.max_control_force, field, 0.1)
    msg.cartesian_control_mode_limits.stop_on_max_control_force = False

    # Joint path execution params
    msg.joint_path_execution_params.joint_relative_velocity = 0.1
    msg.joint_path_execution_params.joint_relative_acceleration = 0.1
    msg.joint_path_execution_params.override_joint_acceleration = 0.0

    # Cartesian path execution params
    for field in cartesian_field_names:
        setattr(msg.cartesian_path_execution_params.max_velocity, field, 0.1)
        setattr(msg.cartesian_path_execution_params.max_acceleration, field, 0.1)
    msg.cartesian_path_execution_params.max_nullspace_velocity = 0.1
    msg.cartesian_path_execution_params.max_nullspace_acceleration = 0.1

    return msg


class MinimalFakeArmInterface:
    def __init__(self,
                 arm_name,
                 control_mode_status_topic,
                 get_control_mode_service_topic,
                 set_control_mode_service_topic):

        self.input_mtx = Lock()

        self.control_mode_parameters_status_msg = DefaultControlModeParametersStatus()
        self.get_control_mode_server =  rospy.Service(get_control_mode_service_topic,   GetControlMode,         self.get_control_mode_service_callback)
        self.set_control_mode_server =  rospy.Service(set_control_mode_service_topic,   SetControlMode,         self.set_control_mode_service_callback)
        self.control_status_pub =       rospy.Publisher(control_mode_status_topic,      ControlModeParameters,  queue_size=1)
        self.control_mode_thread =      Thread(target=self.control_mode_feedback_thread)

    def get_control_mode_service_callback(self, req):
        """
        :type req: GetControlModeRequest
        :return:
        """
        with self.input_mtx:
            return GetControlModeResponse(active_control_mode=self.control_mode_parameters_status_msg, has_active_control_mode=True)

    def set_control_mode_service_callback(self, req):
        """
        :param req: SetControlModeRequest
        :return:
        """
        with self.input_mtx:
            self.control_mode_parameters_status_msg = req.new_control_mode
            return SetControlModeResponse(success=True, message="")

    # TODO: populate seq
    # TODO: populate timestamp in submessages
    def control_mode_feedback_thread(self):
        r = rospy.Rate(control_mode_feedback_rate)
        while not rospy.is_shutdown():
            with self.input_mtx:
                self.control_mode_parameters_status_msg.header.stamp = rospy.Time.now()
                self.control_status_pub.publish(self.control_mode_parameters_status_msg)
            r.sleep()

    def start_feedback_threads(self):
        self.control_mode_thread.start()

    def join_feedback_threads(self):
        self.control_mode_thread.join()


if __name__ == "__main__":
    rospy.init_node("minimal_fake_control_mode_interface")

    interfaces = {}

    for arm in arm_names:
        interfaces[arm] = MinimalFakeArmInterface(
            arm_name=arm,
            control_mode_status_topic=arm + "/control_mode_status",
            get_control_mode_service_topic=arm + "/get_control_mode_service",
            set_control_mode_service_topic=arm + "/set_control_mode_service")

    for arm in arm_names:
        interfaces[arm].start_feedback_threads()

    rospy.loginfo("Publishing data...")
    rospy.spin()

    for arm in arm_names:
        interfaces[arm].join_feedback_threads()
