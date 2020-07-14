#pragma once

#include <peter_msgs/DualGripperTrajectory.h>
#include <peter_msgs/GetDualGripperPoints.h>
#include <peter_msgs/GetObject.h>
#include <peter_msgs/SetDualGripperPoints.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <thread>

namespace gazebo
{
class DualGripperPlugin : public ModelPlugin
{
public:
  ~DualGripperPlugin() override;

  void Load(physics::ModelPtr parent, sdf::ElementPtr sdf) override;

  bool OnAction(peter_msgs::DualGripperTrajectoryRequest &req, peter_msgs::DualGripperTrajectoryResponse &res);

  bool OnGet(peter_msgs::GetDualGripperPointsRequest &req, peter_msgs::GetDualGripperPointsResponse &res);

  bool OnSet(peter_msgs::SetDualGripperPointsRequest &req, peter_msgs::SetDualGripperPointsResponse &res);

private:
  void QueueThread();

  void PrivateQueueThread();

  void OnUpdate();

  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  physics::WorldPtr world_;
  physics::LinkPtr gripper1_;
  physics::LinkPtr gripper2_;

  bool interrupted_{ false };

  std::unique_ptr<ros::NodeHandle> private_ros_node_;
  ros::NodeHandle ros_node_;
  ros::CallbackQueue queue_;
  ros::CallbackQueue private_queue_;
  std::thread ros_queue_thread_;
  std::thread private_ros_queue_thread_;
  ros::ServiceServer action_service_;
  ros::ServiceServer get_service_;
  ros::ServiceServer set_service_;
  ros::Publisher joint_states_pub_;
  ros::Subscriber interrupt_sub_;
  ros::ServiceServer get_gripper1_service_;
  ros::ServiceServer get_gripper2_service_;
};

}  // namespace gazebo
