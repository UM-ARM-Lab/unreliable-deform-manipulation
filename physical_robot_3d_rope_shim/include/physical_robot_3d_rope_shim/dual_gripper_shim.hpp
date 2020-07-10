#ifndef DUAL_GRIPPER_SHIM_HPP
#define DUAL_GRIPPER_SHIM_HPP

#include <memory>

#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <peter_msgs/DualGripperTrajectory.h>
#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "physical_robot_3d_rope_shim/listener.hpp"
#include "physical_robot_3d_rope_shim/planning_interface.hpp"
#include "physical_robot_3d_rope_shim/scene.hpp"

class DualGripperShim
{
public:
  ros::NodeHandle nh_;
  ros::NodeHandle ph_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<PlanningInterace> planner_;
  ros::ServiceServer execute_traj_srv_;
  ros::Publisher talker_;
  using TrajectoryClient = actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>;
  std::unique_ptr<TrajectoryClient> trajectory_client_;
  ros::Duration const traj_goal_time_tolerance_;
  // ros::ServiceClient set_grasping_rope_client_;
  ros::ServiceClient world_control_client_;

  // Manages the planning scene and robot manipulators
  std::shared_ptr<Scene> scene_;

  DualGripperShim(ros::NodeHandle nh, ros::NodeHandle ph);
  void test();
  void gotoHome();
  void settle();

  // Control/exection
  void enableServices();
  bool executeDualGripperTrajectory(peter_msgs::DualGripperTrajectory::Request& req,
                                    peter_msgs::DualGripperTrajectory::Response& res);
  void followJointTrajectory(trajectory_msgs::JointTrajectory const& traj);
};

#endif
