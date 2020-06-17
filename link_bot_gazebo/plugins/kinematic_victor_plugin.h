#pragma once

#include <thread>

#include <peter_msgs/JointTraj.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo {

class KinematicVictorPlugin : public ModelPlugin {
 public:
  ~KinematicVictorPlugin() override;

  void Load(physics::ModelPtr parent, sdf::ElementPtr sdf) override;

  bool OnAction(peter_msgs::JointTrajRequest &req, peter_msgs::JointTrajResponse &res);

 private:
  void QueueThread();

  void PrivateQueueThread();

  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  std::unique_ptr<ros::NodeHandle> private_ros_node_;
  ros::NodeHandle ros_node_;
  ros::CallbackQueue queue_;
  ros::CallbackQueue private_queue_;
  std::thread ros_queue_thread_;
  std::thread private_ros_queue_thread_;
  ros::ServiceServer action_service_;
  physics::JointPtr joint_;
};

}  // namespace gazebo
