#pragma once

#include <link_bot_gazebo/ExecuteAction.h>
#include <link_bot_gazebo/GetObject.h>
#include <link_bot_gazebo/GetObjects.h>
#include <link_bot_gazebo/LinkBotAction.h>
#include <link_bot_gazebo/LinkBotConfiguration.h>
#include <link_bot_gazebo/LinkBotJointConfiguration.h>
#include <link_bot_gazebo/LinkBotPath.h>
#include <link_bot_gazebo/LinkBotReset.h>
#include <link_bot_gazebo/LinkBotState.h>
#include <link_bot_gazebo/LinkBotTrajectory.h>
#include <link_bot_gazebo/NamedPoints.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <ignition/math.hh>
#include <mutex>
#include <sdf/sdf.hh>
#include <string>
#include <thread>

namespace gazebo {

constexpr auto HEAD_Z{0.01};

struct ControlResult {
  link_bot_gazebo::NamedPoints link_bot_config{};
  ignition::math::Vector3d gripper1_vel{ignition::math::Vector3d::Zero};
  ignition::math::Vector3d gripper1_force{ignition::math::Vector3d::Zero};
};

class MultiLinkBotModelPlugin : public ModelPlugin {
 public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

  ~MultiLinkBotModelPlugin() override;

  ControlResult UpdateControl();

  // This is thread safe
  link_bot_gazebo::NamedPoints GetConfiguration();

  void OnUpdate();

  void OnJoy(sensor_msgs::JoyConstPtr msg);

  void OnActionMode(std_msgs::StringConstPtr msg);

  void OnConfiguration(link_bot_gazebo::LinkBotJointConfigurationConstPtr msg);

  bool ExecuteAbsoluteAction(link_bot_gazebo::ExecuteActionRequest &req, link_bot_gazebo::ExecuteActionResponse &res);

  bool ExecuteAction(link_bot_gazebo::ExecuteActionRequest &req, link_bot_gazebo::ExecuteActionResponse &res);

  bool StateServiceCallback(link_bot_gazebo::LinkBotStateRequest &req, link_bot_gazebo::LinkBotStateResponse &res);

  bool GetObjectServiceCallback(link_bot_gazebo::GetObjectRequest &req, link_bot_gazebo::GetObjectResponse &res);

  bool ExecuteTrajectoryCallback(link_bot_gazebo::LinkBotTrajectoryRequest &req,
                                 link_bot_gazebo::LinkBotTrajectoryResponse &res);

  bool LinkBotReset(link_bot_gazebo::LinkBotResetRequest &req, link_bot_gazebo::LinkBotResetResponse &res);

 private:
  auto GetGripper1Pos() -> ignition::math::Vector3d const;
  auto GetGripper1Vel() -> ignition::math::Vector3d const;

  void QueueThread();

  physics::ModelPtr model_;
  event::ConnectionPtr updateConnection_;
  double length_{0.0};
  unsigned int num_links_{0U};
  double kP_pos_{0.0};
  double kI_pos_{0.0};
  double kD_pos_{0.0};
  double kP_vel_{0.0};
  double kI_vel_{0.0};
  double kD_vel_{0.0};
  double max_vel_{1.0};
  double max_speed_{0.15};
  double max_force_{1.0};
  physics::LinkPtr gripper1_link_{nullptr};
  common::PID gripper1_pos_pid_;
  common::PID gripper1_vel_pid_;
  std::mutex control_mutex_;
  ignition::math::Vector3d gripper1_target_position_{0, 0, 0};
  ignition::math::Vector3d gripper1_pos_error_{0, 0, 0};
  ignition::math::Vector3d gripper1_vel_{0, 0, 0};
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::Subscriber joy_sub_;
  ros::Subscriber action_mode_sub_;
  ros::Subscriber config_sub_;
  ros::ServiceServer state_service_;
  ros::ServiceServer get_object_service_;
  ros::ServiceServer execute_action_service_;
  ros::ServiceServer execute_absolute_action_service_;
  ros::ServiceServer execute_traj_service_;
  ros::ServiceServer reset_service_;
  ros::ServiceClient objects_service_;
  ros::Publisher register_link_bot_pub_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  ros::CallbackQueue execute_trajs_queue_;
  std::thread execute_trajs_ros_queue_thread_;
  std::string mode_{"disabled"};
};
}  // namespace gazebo
