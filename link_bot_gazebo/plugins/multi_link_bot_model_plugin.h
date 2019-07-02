#pragma once

#include <string>
#include <thread>

#include <geometry_msgs/Wrench.h>
#include <link_bot_gazebo/LinkBotConfiguration.h>
#include <link_bot_gazebo/LinkBotState.h>
#include <link_bot_gazebo/MultiLinkBotPositionAction.h>
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
class MultiLinkBotModelPlugin : public ModelPlugin {
 public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

  ~MultiLinkBotModelPlugin() override;

  void OnUpdate();

  void OnJoy(sensor_msgs::JoyConstPtr msg);

  void OnAction(link_bot_gazebo::MultiLinkBotPositionActionConstPtr msg);

  void OnConfiguration(link_bot_gazebo::LinkBotConfigurationConstPtr msg);

  bool StateServiceCallback(link_bot_gazebo::LinkBotStateRequest &req, link_bot_gazebo::LinkBotStateResponse &res);

 protected:
 private:
  void QueueThread();

  physics::ModelPtr model_;
  event::ConnectionPtr updateConnection_;
  double kP_{5};
  double kI_{0};
  double kD_{0};
  physics::LinkPtr gripper1_link_{nullptr};
  physics::LinkPtr gripper2_link_{nullptr};
  std::vector<geometry_msgs::Wrench> wrenches_;
  common::PID gripper1_x_pos_pid_;
  common::PID gripper1_y_pos_pid_;
  common::PID gripper2_x_pos_pid_;
  common::PID gripper2_y_pos_pid_;
  ignition::math::Vector3d gripper1_target_position_{0, 0, 0};
  ignition::math::Vector3d gripper2_target_position_{0, 0, 0};
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::Subscriber joy_sub_;
  ros::Subscriber action_sub_;
  ros::Subscriber config_sub_;
  ros::ServiceServer state_service_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
};
}  // namespace gazebo
