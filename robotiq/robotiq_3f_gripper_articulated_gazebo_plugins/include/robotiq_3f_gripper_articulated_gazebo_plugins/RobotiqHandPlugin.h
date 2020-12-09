/*
 * Copyright 2014 Open Source Robotics Foundation
 * Copyright 2015 Clearpath Robotics
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#pragma once

#include <string>
#include <vector>
#include <thread>

#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>

#include <ros/advertise_options.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>

#include <victor_hardware_interface_msgs/Robotiq3FingerCommand.h>
#include <victor_hardware_interface_msgs/Robotiq3FingerStatus.h>

#include <robotiq_3f_gripper_articulated_gazebo_plugins/robotiq_control.h>

/// \brief A plugin that implements the Robotiq 3-Finger Adaptative Gripper.
/// The plugin exposes the next parameters via SDF tags:
///   * <side> Determines if we are controlling the left or right hand. This is
///            a required parameter and the allowed values are 'left' or 'right'
///   * <kp_position> P gain for the PID that controls the position of the joints.
///   * <ki_position> I gain for the PID that controls the position of the joints.
///   * <kd_position> D gain for the PID that controls the position of the joints.
///   * <topic_command> ROS topic name used to send new commands to the hand. This parameter is optional.
///   * <topic_status> ROS topic name used to receive state from the hand. This parameter is optional.
class RobotiqHandPlugin : public gazebo::ModelPlugin
{

 public:
  ~RobotiqHandPlugin() override;

  void Load(gazebo::physics::ModelPtr _parent, sdf::ElementPtr _sdf) override;

  void QueueThread();

  void OnGripperCommand(victor_hardware_interface_msgs::Robotiq3FingerCommandConstPtr const &msg);

  std::unique_ptr<ros::NodeHandle> ph_;

  ros::Subscriber command_sub_;
  ros::Publisher status_pub_;

  /// \brief ROS callback queue.
  ros::CallbackQueue rosQueue;

  /// \brief ROS callback queue thread.
  std::thread ros_queue_thread_;

  /// \brief gazebo world update connection.
  gazebo::event::ConnectionPtr updateConnection;

  std::unique_ptr<RobotiqControl> control_;
};

template<typename T>
T get(sdf::ElementPtr sdf, std::string key, T default_value)
{
  if (sdf->HasElement(key))
  {
    return sdf->GetElement(key)->Get<T>();
  }
  return default_value;
}