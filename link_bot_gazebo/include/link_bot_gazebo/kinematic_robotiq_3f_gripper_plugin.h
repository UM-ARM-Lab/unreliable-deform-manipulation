#pragma once

#include <memory>

#include <victor_hardware_interface_msgs/Robotiq3FingerCommand.h>
#include <victor_hardware_interface_msgs/Robotiq3FingerStatus.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>

namespace gazebo
{
class KinematicRobotiq3fGripperPlugin : public ModelPlugin
{
 public:
  ~KinematicRobotiq3fGripperPlugin() override;

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override;

  void OnCommand(victor_hardware_interface_msgs::Robotiq3FingerCommandConstPtr const &msg);

 private:
  void QueueThread();

  void PeriodicUpdate();

  void CreateServices();

  void PrivateQueueThread();

  void OnUpdate();

  physics::ModelPtr model_;
  event::ConnectionPtr update_connection_;
  std::unique_ptr<ros::NodeHandle> private_ros_node_;
  ros::NodeHandle ros_node_;
  ros::CallbackQueue queue_;
  ros::CallbackQueue private_queue_;
  std::thread ros_queue_thread_;
  std::thread private_ros_queue_thread_;
  ros::Subscriber command_sub_;
  ros::Publisher status_pub_;
  ros::Publisher joint_state_pub_;
  std::thread periodic_event_thread_;
  std::string arm_name_;

  physics::JointPtr finger_a_joint_;
  physics::JointPtr finger_b_joint_;
  physics::JointPtr finger_c_joint_; // middle
  double status_rate_ = 50.0;
  std::string robot_namespace_;
  std::string prefix_;
  std::string action_name_;
};

}  // namespace gazebo
