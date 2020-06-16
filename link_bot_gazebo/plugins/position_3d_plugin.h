#pragma once

#include <geometry_msgs/Pose.h>
#include <peter_msgs/ActionSpaceDescription.h>
#include <peter_msgs/GetObject.h>
#include <peter_msgs/GetPosition3D.h>
#include <peter_msgs/ModelsEnable.h>
#include <peter_msgs/ModelsPoses.h>
#include <peter_msgs/Position3DAction.h>
#include <peter_msgs/Position3DEnable.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>

#include <functional>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>

namespace gazebo {

class Position3dPlugin : public ModelPlugin {
 public:
  ~Position3dPlugin() override;

  void Load(physics::ModelPtr parent, sdf::ElementPtr sdf) override;

  void OnUpdate(common::UpdateInfo const &info);

  bool OnStop(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res);

  bool OnEnable(peter_msgs::Position3DEnableRequest &req, peter_msgs::Position3DEnableResponse &res);

  bool OnAction(peter_msgs::Position3DActionRequest &req, peter_msgs::Position3DActionResponse &res);

  bool GetPos(peter_msgs::GetPosition3DRequest &req, peter_msgs::GetPosition3DResponse &res);

  bool GetActionSpace(peter_msgs::ActionSpaceDescriptionRequest &req, peter_msgs::ActionSpaceDescriptionResponse &res);

  bool GetObjectCallback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res);

 private:
  void QueueThread();

  void PrivateQueueThread();

  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  physics::LinkPtr link_;
  physics::CollisionPtr collision_;
  std::string link_name_;
  bool enabled_{true};
  std::unique_ptr<ros::NodeHandle> private_ros_node_;
  ros::NodeHandle ros_node_;
  ros::CallbackQueue queue_;
  ros::CallbackQueue private_queue_;
  std::thread ros_queue_thread_;
  std::thread private_ros_queue_thread_;
  ros::ServiceServer enable_service_;
  ros::ServiceServer action_service_;
  ros::ServiceServer stop_service_;
  ros::ServiceServer get_position_service_;
  ros::ServiceServer action_space_service_;
  ros::ServiceServer get_object_service_;
  ros::Publisher register_object_pub_;
  double kP_pos_{0.0};
  double kD_pos_{0.0};
  double max_vel_{0.0};
  double kP_vel_{0.0};
  double kI_vel_{0.0};
  double kD_vel_{0.0};
  double kP_rot_{0.0};
  double kD_rot_{0.0};
  double max_rot_vel_{0.0};
  double kP_rot_vel_{0.0};
  double kD_rot_vel_{0.0};
  double max_torque_{0.0};
  double max_force_{0.0};
  common::PID pos_pid_;
  common::PID vel_pid_;
  common::PID rot_pid_;
  common::PID rot_vel_pid_;
  ignition::math::Vector3d target_position_{0, 0, 0};
  ignition::math::Vector3d pos_error_{0, 0, 0};
  double rot_error_{0};
  double total_mass_{0.0};
  std::string name_;
  bool gravity_compensation_{false};
  double z_integral_{0.0};
};

}  // namespace gazebo
