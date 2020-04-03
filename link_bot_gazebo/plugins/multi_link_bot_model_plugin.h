#pragma once

#include <peter_msgs/ExecuteAction.h>
#include <peter_msgs/GetObject.h>
#include <peter_msgs/GetObjects.h>
#include <peter_msgs/LinkBotReset.h>
#include <peter_msgs/LinkBotState.h>
#include <peter_msgs/NamedPoints.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>

#include <Eigen/Eigen>
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

struct ControlResult {
  ignition::math::Vector3d gripper1_force{ignition::math::Vector3d::Zero};
};

class MultiLinkBotModelPlugin : public ModelPlugin {
 public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

  ~MultiLinkBotModelPlugin() override;

  ControlResult UpdateControl();

  // This is thread safe
  peter_msgs::NamedPoints GetConfiguration();

  void OnUpdate();

  void OnJoy(sensor_msgs::JoyConstPtr msg);

  void OnActionMode(std_msgs::StringConstPtr msg);

  bool ExecuteAbsoluteAction(peter_msgs::ExecuteActionRequest &req, peter_msgs::ExecuteActionResponse &res);

  bool ExecuteAction(peter_msgs::ExecuteActionRequest &req, peter_msgs::ExecuteActionResponse &res);

  bool StateServiceCallback(peter_msgs::LinkBotStateRequest &req, peter_msgs::LinkBotStateResponse &res);

  bool GetObjectGripperCallback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res);

  bool GetObjectLinkBotCallback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res);

  bool ResetRobot(peter_msgs::LinkBotResetRequest &req, peter_msgs::LinkBotResetResponse &res);

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
  ros::NodeHandle ros_node_;
  ros::Subscriber joy_sub_;
  ros::Subscriber action_mode_sub_;
  ros::Subscriber config_sub_;
  ros::ServiceServer state_service_;
  ros::ServiceServer get_object_link_bot_service_;
  ros::ServiceServer get_object_gripper_service_;
  ros::ServiceServer execute_action_service_;
  ros::ServiceServer execute_absolute_action_service_;
  ros::ServiceServer execute_traj_service_;
  ros::ServiceServer reset_service_;
  ros::ServiceClient objects_service_;
  ros::Publisher register_object_pub_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  ros::CallbackQueue execute_trajs_queue_;
  std::thread execute_trajs_ros_queue_thread_;
  std::string mode_{"disabled"};

  // these allow one to make the gripper have some arbitrary linear dynamics
  Eigen::Matrix2d A_{Eigen::Matrix2d::Identity()};
  Eigen::Matrix2d B_{Eigen::Matrix2d::Identity()};
};
}  // namespace gazebo
