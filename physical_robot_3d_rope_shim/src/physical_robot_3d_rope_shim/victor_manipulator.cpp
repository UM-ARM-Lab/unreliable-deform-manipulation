//
// Created by arprice on 9/13/18.
//

#include "physical_robot_3d_rope_shim/victor_manipulator.hpp"
#include <victor_hardware_interface/SetControlMode.h>
#include <arc_utilities/ros_helpers.hpp>

namespace hal = victor_hardware_interface;

std::vector<double> getQHome(ros::NodeHandle& nh, robot_model::JointModelGroup const* const arm)
{
  auto const defaultHome = [&arm]() {
	if (arm->getName().find("left") != std::string::npos)
	{
	  return std::vector<double>{ -M_PI_2, M_PI_2, 0, 0, 0, 0, 0 };
	}
	else if (arm->getName().find("right") != std::string::npos)
	{
	  return std::vector<double>{ -M_PI_2, -M_PI_2, 0, 0, 0, 0, 0 };
	}
	else
	{
	  throw std::runtime_error("Unknown arm.");
	}
  }();

  return ROSHelpers::GetVector<double>(nh, arm->getName() + "_home", defaultHome);
}

VictorManipulator::VictorManipulator(ros::NodeHandle& nh,
									                   robot_model::RobotModelPtr _pModel,
									                   robot_model::JointModelGroup* _arm,
									                   robot_model::JointModelGroup* _gripper,
									                   std::string _palmName)
  : Manipulator(_pModel, _arm, _gripper, _palmName), qHome(getQHome(nh, _arm))
{
  if (arm->getName().find("left") == std::string::npos &&
  	  arm->getName().find("right") == std::string::npos)
  {
	  throw std::runtime_error("Unknown arm.");
  }

  const std::string serviceName = arm->getName() + "/set_control_mode_service";
  configurationClient = nh.serviceClient<hal::SetControlMode>(serviceName);
  if (!configurationClient.waitForExistence(ros::Duration(3)))
  {
	  ROS_WARN("Robot configuration server not connected.");
  }
}

bool VictorManipulator::configureHardware()
{
  ROS_INFO_STREAM("Configuring " << arm->getName() << " to position mode.");

  hal::SetControlModeRequest req;
  req.new_control_mode.control_mode.mode = hal::ControlMode::JOINT_POSITION;
  req.new_control_mode.joint_path_execution_params.joint_relative_velocity = 0.8;
  req.new_control_mode.joint_path_execution_params.joint_relative_acceleration = 0.8;

  // req.new_control_mode.control_mode.mode = hal::ControlMode::JOINT_IMPEDANCE;
  // req.new_control_mode.joint_path_execution_params.joint_relative_velocity = 0.8;
  // req.new_control_mode.joint_path_execution_params.joint_relative_acceleration = 0.8;

  // const double damping = 0.7;
  // req.new_control_mode.joint_impedance_params.joint_damping.joint_1 = damping;
  // req.new_control_mode.joint_impedance_params.joint_damping.joint_2 = damping;
  // req.new_control_mode.joint_impedance_params.joint_damping.joint_3 = damping;
  // req.new_control_mode.joint_impedance_params.joint_damping.joint_4 = damping;
  // req.new_control_mode.joint_impedance_params.joint_damping.joint_5 = damping;
  // req.new_control_mode.joint_impedance_params.joint_damping.joint_6 = damping;
  // req.new_control_mode.joint_impedance_params.joint_damping.joint_7 = damping;

  // req.new_control_mode.joint_impedance_params.joint_stiffness.joint_1 = 100.0;
  // req.new_control_mode.joint_impedance_params.joint_stiffness.joint_2 = 100.0;
  // req.new_control_mode.joint_impedance_params.joint_stiffness.joint_3 = 50.0;
  // req.new_control_mode.joint_impedance_params.joint_stiffness.joint_4 = 50.0;
  // req.new_control_mode.joint_impedance_params.joint_stiffness.joint_5 = 50.0;
  // req.new_control_mode.joint_impedance_params.joint_stiffness.joint_6 = 150.0;
  // req.new_control_mode.joint_impedance_params.joint_stiffness.joint_7 = 200.0;

  req.new_control_mode.header.frame_id = "victor_root";
  req.new_control_mode.header.stamp = ros::Time::now();

  hal::SetControlModeResponse resp;
  bool callSucceeded = configurationClient.call(req, resp);
  return callSucceeded && resp.success;
}

std::vector<double> VictorManipulator::getGripperOpenJoints() const
{
  return std::vector<double>(gripper->getVariableCount(), 0.0);
}
