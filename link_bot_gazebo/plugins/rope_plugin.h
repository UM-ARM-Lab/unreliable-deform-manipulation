#pragma once

#include <peter_msgs/ExecuteAction.h>
#include <peter_msgs/GetObject.h>
#include <peter_msgs/GetObjects.h>
#include <peter_msgs/LinkBotState.h>
#include <peter_msgs/NamedPoints.h>
#include <peter_msgs/SetRopeConfiguration.h>
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

class MultiLinkBotModelPlugin : public ModelPlugin {
 public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

  ~MultiLinkBotModelPlugin() override;

  bool StateServiceCallback(peter_msgs::LinkBotStateRequest &req, peter_msgs::LinkBotStateResponse &res);

  bool GetObjectLinkBotCallback(peter_msgs::GetObjectRequest &req, peter_msgs::GetObjectResponse &res);

  bool SetRopeConfigCallback(peter_msgs::SetRopeConfigurationRequest &req,
                             peter_msgs::SetRopeConfigurationResponse &res);

 private:
  void QueueThread();

  physics::ModelPtr model_;
  event::ConnectionPtr updateConnection_;
  double length_{0.0};
  unsigned int num_links_{0U};
  ros::NodeHandle ros_node_;
  ros::ServiceServer state_service_;
  ros::ServiceServer set_configuration_service_;
  ros::ServiceServer get_object_link_bot_service_;
  ros::ServiceClient objects_service_;
  ros::Publisher register_object_pub_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
};
}  // namespace gazebo
