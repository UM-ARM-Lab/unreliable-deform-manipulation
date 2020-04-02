#pragma once
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <sdf/sdf.hh>

#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>

using namespace gazebo;

class CarPlugin : public ModelPlugin {
 public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf);

  void Update(const common::UpdateInfo &info);

 private:
  void QueueThread();


  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  ros::ServiceServer state_service_;
  ros::ServiceServer execute_action_service_;
  ros::Publisher register_car_pub_;
  physics::ModelPtr model_;
  physics::LinkPtr body_;
  physics::JointPtr right_wheel_;
  physics::JointPtr left_wheel_;
  common::PID left_wheel_vel_pid_;
  common::PID right_wheel_vel_pid_;
};
