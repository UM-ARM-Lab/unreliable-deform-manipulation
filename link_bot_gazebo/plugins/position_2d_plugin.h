#pragma once

#include <ros/ros.h>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>

#include <geometry_msgs/Pose.h>
#include <link_bot_gazebo/Position2dAction.h>
#include <link_bot_gazebo/Position2dEnable.h>
#include <std_msgs/Empty.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

namespace gazebo {

class Position2dPlugin : public ModelPlugin {
 public:
  ~Position2dPlugin() override;

  void Load(physics::ModelPtr parent, sdf::ElementPtr sdf) override;

  void OnUpdate();

  void OnStop(std_msgs::EmptyConstPtr msg);

  void OnEnable(link_bot_gazebo::Position2dEnableConstPtr msg);

  void OnAction(link_bot_gazebo::Position2dActionConstPtr msg);

 private:
  void QueueThread();

  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  physics::LinkPtr link_;
  std::string link_name_;
  bool enabled_{true};
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  ros::Subscriber enable_sub_;
  ros::Subscriber action_sub_;
  ros::Subscriber stop_sub_;
  double kP_pos_{0.0};
  double kI_pos_{0.0};
  double kD_pos_{0.0};
  double max_vel_{0.0};
  double kP_rot_{0.0};
  double kI_rot_{0.0};
  double kD_rot_{0.0};
  double max_torque_{0.0};
  double kP_vel_{0.0};
  double kI_vel_{0.0};
  double kD_vel_{0.0};
  double max_force_{0.0};
  common::PID x_pos_pid_;
  common::PID y_pos_pid_;
  common::PID z_rot_pid_;
  common::PID x_vel_pid_;
  common::PID y_vel_pid_;
  ignition::math::Pose3d target_pose_{0, 0, 0, 0, 0, 0};
  ignition::math::Vector3d target_velocity_{0, 0, 0};
};

}  // namespace gazebo
