#ifndef VICTOR_SHIM_HPP
#define VICTOR_SHIM_HPPP

#include <peter_msgs/DualGripperTrajectory.h>
#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <memory>

#include "victor_3d_rope_shim/victor_interface.h"

class VictorShim
{
public:
  ros::NodeHandle nh_;
  ros::NodeHandle ph_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<VictorInterface> victor_;
  ros::ServiceServer execute_traj_srv_;

  VictorShim(ros::NodeHandle nh, ros::NodeHandle ph);

  // Victor control/exection
  void enableServices();
  bool executeTrajectory(peter_msgs::DualGripperTrajectory::Request& req,
                         peter_msgs::DualGripperTrajectory::Response& res);
};

#endif
