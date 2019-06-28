#pragma once

#include <string>
#include <thread>

#include <link_bot_gazebo/LinkBotConfiguration.h>
#include <link_bot_gazebo/LinkBotForceAction.h>
#include <link_bot_gazebo/LinkBotVelocityAction.h>
#include <link_bot_gazebo/LinkBotState.h>
#include <geometry_msgs/Wrench.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <sensor_msgs/Joy.h>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <ignition/math.hh>
#include <sdf/sdf.hh>

namespace gazebo {
class LinkBotModelPlugin : public ModelPlugin {
 public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

  ~LinkBotModelPlugin() override;

  void OnUpdate();

  void OnJoy(sensor_msgs::JoyConstPtr msg);

  void OnForceAction(link_bot_gazebo::LinkBotForceActionConstPtr msg);

  void OnVelocityAction(link_bot_gazebo::LinkBotVelocityActionConstPtr msg);

  void OnConfiguration(link_bot_gazebo::LinkBotConfigurationConstPtr msg);

  bool StateServiceCallback(link_bot_gazebo::LinkBotStateRequest &req, link_bot_gazebo::LinkBotStateResponse &res);

 protected:
 private:
  void QueueThread();

  physics::ModelPtr model_;
  event::ConnectionPtr updateConnection_;
  sensors::ContactSensorPtr contact_sensor_;
  double kP_{500};
  double kI_{0};
  double kD_{0};
  bool use_force_{false};
  physics::LinkPtr velocity_control_link_{nullptr};
  std::vector<geometry_msgs::Wrench> wrenches_;
  common::PID x_vel_pid_;
  common::PID y_vel_pid_;
  ignition::math::Vector3d target_linear_vel_{0, 0, 0};
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::Subscriber joy_sub_;
  ros::Subscriber vel_cmd_sub_;
  ros::Subscriber force_cmd_sub_;
  ros::Subscriber config_sub_;
  ros::ServiceServer state_service_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  double action_scale{1.0};
};
}  // namespace gazebo
