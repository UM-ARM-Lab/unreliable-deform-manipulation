#pragma once

#include <actionlib/server/simple_action_server.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <thread>

namespace gazebo {

using TrajServer = actionlib::SimpleActionServer<control_msgs::FollowJointTrajectoryAction>;

class KinematicVictorPlugin : public ModelPlugin {
 public:
  ~KinematicVictorPlugin() override;

  void Load(physics::ModelPtr parent, sdf::ElementPtr sdf) override;

  void FollowJointTrajectory(const TrajServer::GoalConstPtr &goal);

 private:
  void QueueThread();

  void PrivateQueueThread();

  void OnUpdate();

  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  physics::WorldPtr world_;

  std::unique_ptr<ros::NodeHandle> private_ros_node_;
  ros::NodeHandle ros_node_;
  ros::CallbackQueue queue_;
  ros::CallbackQueue private_queue_;
  std::thread ros_queue_thread_;
  std::thread private_ros_queue_thread_;
  ros::Publisher joint_states_pub_;

  std::unique_ptr<TrajServer> follow_traj_server_;
};

}  // namespace gazebo
