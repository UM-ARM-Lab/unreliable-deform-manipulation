#pragma once

#include <geometry_msgs/Pose.h>
#include <peter_msgs/GetObject.h>
#include <peter_msgs/LinkBotState.h>
#include <peter_msgs/LinkBotTrajectory.h>
#include <peter_msgs/ModelsEnable.h>
#include <peter_msgs/ModelsPoses.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>

namespace gazebo {

// FIXME: implement copy/assign/move
class TetherPlugin : public ModelPlugin {
 public:
  ~TetherPlugin() override;

  void Load(physics::ModelPtr parent, sdf::ElementPtr sdf) override;

  bool GetObjectServiceCallback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res);

 private:
  void QueueThread();

  event::ConnectionPtr update_connection_;
  physics::ModelPtr model_;
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  ros::ServiceServer state_service_;
  ros::Publisher register_tether_pub_;
  unsigned int num_links_{0u};
};

}  // namespace gazebo
