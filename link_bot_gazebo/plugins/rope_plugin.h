#pragma once

#include <peter_msgs/GetBool.h>
#include <peter_msgs/GetRopeState.h>
#include <peter_msgs/SetRopeState.h>
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

namespace gazebo
{
class RopePlugin : public ModelPlugin
{
public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

  ~RopePlugin() override;

  bool SetRopeState(peter_msgs::SetRopeStateRequest &req, peter_msgs::SetRopeStateResponse &res);

  bool GetRopeState(peter_msgs::GetRopeStateRequest &req, peter_msgs::GetRopeStateResponse &res);

  bool GetOverstretched(peter_msgs::GetBoolRequest &req, peter_msgs::GetBoolResponse &res);

private:
  void QueueThread();

  physics::ModelPtr model_;
  physics::LinkPtr rope_link1_;
  physics::LinkPtr gripper1_;
  physics::LinkPtr gripper2_;
  event::ConnectionPtr updateConnection_;
  double length_{ 0.0 };
  double rest_distance_between_gripper1_and_link_1_{ 0.0 };
  unsigned int num_links_{ 0U };
  ros::NodeHandle ros_node_;
  ros::ServiceServer set_state_service_;
  ros::ServiceServer rope_overstretched_service_;
  ros::ServiceServer get_state_service_;
  ros::ServiceServer get_object_link_bot_service_;
  ros::Publisher register_object_pub_;
  ros::CallbackQueue queue_;
  std::thread ros_queue_thread_;
  double overstretching_factor_{ 1.0 };
};
}  // namespace gazebo
