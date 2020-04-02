#pragma once

#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <sdf/sdf.hh>

#include <peter_msgs/ExecuteAction.h>
#include <peter_msgs/GetObject.h>
#include <peter_msgs/LinkBotReset.h>

using namespace gazebo;

class CarPlugin : public ModelPlugin {
 public:
  ~CarPlugin() override;

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf);

  void Update(const common::UpdateInfo &info);

  bool ExecuteAction(peter_msgs::ExecuteActionRequest &req, peter_msgs::ExecuteActionResponse &res);

  void UpdateControl();

  bool GetObjectCarCallback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res);

  bool ResetRobot(peter_msgs::LinkBotResetRequest &req, peter_msgs::LinkBotResetResponse &res);

 private:
  void QueueThread();

  std::mutex control_mutex_;
  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  ros::NodeHandle ros_node_;
  ros::CallbackQueue queue_;
  ros::ServiceClient objects_service_;
  std::thread ros_queue_thread_;
  ros::ServiceServer state_service_;
  ros::ServiceServer execute_action_service_;
  ros::ServiceServer reset_service_;
  ros::ServiceServer get_object_car_service_;
  ros::Publisher register_car_pub_;
  ros::Publisher wheel_seed_pub_;
  physics::LinkPtr body_;
  physics::JointPtr right_wheel_;
  physics::JointPtr left_wheel_;
  common::PID left_wheel_vel_pid_;
  common::PID right_wheel_vel_pid_;

  double kP_vel_{0.0};
  double kI_vel_{0.0};
  double kD_vel_{0.0};
  double kFF_vel_{0.0};
  double max_force_{1.0};
  double max_speed_{5};
  double left_force_{0.0};
  double right_force_{0.0};
  double left_wheel_target_velocity_{0.0};
  double right_wheel_target_velocity_{0.0};
};
