#pragma once

#include <mutex>
#include <string>
#include <thread>

#include <link_bot_gazebo/LinkBotJointConfiguration.h>
#include <link_bot_gazebo/LinkBotConfiguration.h>
#include <link_bot_gazebo/LinkBotPath.h>
#include <link_bot_gazebo/LinkBotPositionAction.h>
#include <link_bot_gazebo/LinkBotState.h>
#include <link_bot_gazebo/LinkBotTrajectory.h>
#include <link_bot_gazebo/LinkBotVelocityAction.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/String.h>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <ignition/math.hh>
#include <sdf/sdf.hh>

namespace gazebo {

constexpr auto HEAD_Z{0.01};

struct ControlResult {
  link_bot_gazebo::LinkBotConfiguration configuration{};
  ignition::math::Vector3d gripper1_vel{ignition::math::Vector3d::Zero};
  ignition::math::Vector3d gripper2_vel{ignition::math::Vector3d::Zero};
  ignition::math::Vector3d gripper1_force{ignition::math::Vector3d::Zero};
  ignition::math::Vector3d gripper2_force{ignition::math::Vector3d::Zero};
};

class MultiLinkBotModelPlugin : public ModelPlugin {
 public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

  ~MultiLinkBotModelPlugin() override;

  ControlResult UpdateControl();

  // This is thread safe
  link_bot_gazebo::LinkBotConfiguration GetConfiguration();

  void OnUpdate();

  void OnPostRender();

  void OnJoy(sensor_msgs::JoyConstPtr msg);

  bool OnPositionAction(link_bot_gazebo::LinkBotPositionActionRequest &req,
                        link_bot_gazebo::LinkBotPositionActionResponse &res);

  void OnVelocityAction(link_bot_gazebo::LinkBotVelocityActionConstPtr msg);

  void OnActionMode(std_msgs::StringConstPtr msg);

  void OnConfiguration(link_bot_gazebo::LinkBotJointConfigurationConstPtr msg);

  bool StateServiceCallback(link_bot_gazebo::LinkBotStateRequest &req, link_bot_gazebo::LinkBotStateResponse &res);

  bool ExecutePathCallback(link_bot_gazebo::LinkBotPathRequest &req, link_bot_gazebo::LinkBotPathResponse &res);

  bool ExecuteTrajectoryCallback(link_bot_gazebo::LinkBotTrajectoryRequest &req,
                                 link_bot_gazebo::LinkBotTrajectoryResponse &res);

 private:
  void QueueThread();

  physics::ModelPtr model_;
  sensors::CameraSensorPtr camera_sensor;
  event::ConnectionPtr updateConnection_;
  event::ConnectionPtr postRenderConnection_;
  uint32_t image_sequence_number{0u};
  sensor_msgs::Image latest_image_;
  bool ready_{false};
  double length_{0u};
  unsigned int num_links_{0u};
  double kP_pos_{0.0};
  double kI_pos_{0.0};
  double kD_pos_{0.0};
  double kP_vel_{0.0};
  double kI_vel_{0.0};
  double kD_vel_{0.0};
  double max_vel_{1.0};
  double max_force_{1.0};
  double max_vel_acc_{1.0};
  physics::LinkPtr gripper1_link_{nullptr};
  physics::LinkPtr gripper2_link_{nullptr};
  common::PID gripper1_x_pos_pid_;
  common::PID gripper1_y_pos_pid_;
  common::PID gripper2_x_pos_pid_;
  common::PID gripper2_y_pos_pid_;
  common::PID gripper1_x_vel_pid_;
  common::PID gripper1_y_vel_pid_;
  common::PID gripper2_x_vel_pid_;
  common::PID gripper2_y_vel_pid_;
  std::mutex control_mutex_;
  ignition::math::Vector3d gripper1_target_position_{0, 0, 0};
  ignition::math::Vector3d gripper2_target_position_{0, 0, 0};
  ignition::math::Vector3d gripper1_pos_error_{0, 0, 0};
  ignition::math::Vector3d gripper2_pos_error_{0, 0, 0};
  ignition::math::Vector3d gripper1_target_velocity_{0, 0, 0};
  ignition::math::Vector3d gripper1_current_target_velocity_{0, 0, 0};
  // TODO: implement acceleration for gripper 2
  ignition::math::Vector3d gripper2_target_velocity_{0, 0, 0};
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::Subscriber joy_sub_;
  ros::ServiceServer pos_action_service_;
  ros::Subscriber velocity_action_sub_;
  ros::Subscriber action_mode_sub_;
  ros::Subscriber config_sub_;
  ros::ServiceServer state_service_;
  ros::ServiceServer execute_path_service_;
  ros::ServiceServer execute_traj_service_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  std::string mode_{"velocity"};
};
}  // namespace gazebo
