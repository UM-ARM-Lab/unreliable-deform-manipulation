#ifndef DUAL_GRIPPER_SHIM_HPP
#define DUAL_GRIPPER_SHIM_HPP

#include <memory>

#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <peter_msgs/DualGripperTrajectory.h>

#include "physical_robot_3d_rope_shim/listener.hpp"
#include "physical_robot_3d_rope_shim/scene.hpp"
#include "physical_robot_3d_rope_shim/planning_interface.hpp"

class DualGripperShim
{
public:
  ros::NodeHandle nh_;
  ros::NodeHandle ph_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<PlanningInterace> robot_;
  ros::ServiceServer execute_traj_srv_;

  // Manages the planning scene and robot manipulators
  std::shared_ptr<Scene> scene_;

  DualGripperShim(ros::NodeHandle nh, ros::NodeHandle ph);
  void test();

  // Control/exection
  void enableServices();
  bool executeTrajectory(peter_msgs::DualGripperTrajectory::Request& req,
                         peter_msgs::DualGripperTrajectory::Response& res);
};

#endif
