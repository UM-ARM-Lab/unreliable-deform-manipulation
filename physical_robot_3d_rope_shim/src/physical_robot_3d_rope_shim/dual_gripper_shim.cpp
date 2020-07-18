#include "physical_robot_3d_rope_shim/dual_gripper_shim.hpp"
#include "physical_robot_3d_rope_shim/val_interface.hpp"
#include "physical_robot_3d_rope_shim/victor_interface.hpp"

#include <peter_msgs/SetBool.h>
#include <peter_msgs/WorldControl.h>
#include <std_msgs/String.h>
#include <arc_utilities/ros_helpers.hpp>

#include "assert.hpp"
#include "eigen_ros_conversions.hpp"

namespace pm = peter_msgs;

#define COLOR_GREEN "\033[32m"
#define COLOR_NORMAL "\033[0m"

DualGripperShim::DualGripperShim(ros::NodeHandle nh, ros::NodeHandle ph)
  : nh_(nh)
  , ph_(ph)
  , tf_buffer_(std::make_shared<tf2_ros::Buffer>())
  , tf_listener_(std::make_shared<tf2_ros::TransformListener>(*tf_buffer_))
  , talker_(nh_.advertise<std_msgs::String>("polly", 10, false))
  , trajectory_client_(std::make_unique<TrajectoryClient>("both_arms_controller/follow_joint_trajectory", true))
  , traj_goal_time_tolerance_(ROSHelpers::GetParam(ph_, "traj_goal_time_tolerance", 0.05))
  , set_grasping_rope_client_(nh_.serviceClient<peter_msgs::SetBool>("set_grasping_rope"))
  , world_control_client_(nh_.serviceClient<peter_msgs::WorldControl>("world_control"))
{
  auto const robot_name = ROSHelpers::GetParam<std::string>(nh, "robot_name", "val");
  assert(robot_name == "val" || robot_name == "victor");

  if (robot_name == "victor")
  {
    planner_ = std::make_shared<VictorInterface>(nh, ph, tf_buffer_, "both_arms");
  }
  else if (robot_name == "val")
  {
    planner_ = std::make_shared<ValInterface>(nh_, ph_, tf_buffer_, "both_arms");
  }
  planner_->configureHomeState();
  scene_ = std::make_shared<Scene>(nh_, ph_, planner_);
}

////////////////////////////////////////////////////////////////////////////////

void DualGripperShim::gotoHome()
{
  ROS_INFO("Going home");
  // let go of the rope
  peter_msgs::SetBool release_rope;
  release_rope.request.data = false;
  set_grasping_rope_client_.call(release_rope);

  // TODO: make this a service call
  auto ps = scene_->clonePlanningScene();
  ROS_INFO("Planning to home");
  auto const traj = planner_->plan(ps, planner_->home_state_);
  followJointTrajectory(traj);
  ROS_INFO("Done attempting to move home");

  peter_msgs::SetBool grasp_rope;
  grasp_rope.request.data = true;
  set_grasping_rope_client_.call(grasp_rope);

  settle();
}

bool DualGripperShim::gotoHomeCallback(std_srvs::EmptyRequest& /*req*/, std_srvs::EmptyResponse& /*res*/)
{
  gotoHome();
  return true;
}

void DualGripperShim::settle()
{
  peter_msgs::WorldControl settle;
  settle.request.seconds = 10;
  world_control_client_.call(settle);
}

void DualGripperShim::enableServices()
{
  execute_traj_srv_ =
      nh_.advertiseService("execute_dual_gripper_action", &DualGripperShim::executeDualGripperTrajectory, this);
  goto_home_srv_ = nh_.advertiseService("goto_home", &DualGripperShim::gotoHomeCallback, this);
  ROS_WARN_STREAM(COLOR_GREEN << "Ready for commands" << COLOR_NORMAL);
}

bool DualGripperShim::executeDualGripperTrajectory(pm::DualGripperTrajectory::Request& req,
                                                   pm::DualGripperTrajectory::Response& res)
{
  if (req.gripper1_points.size() != req.gripper2_points.size())
  {
    ROS_WARN("Mismatched gripper trajectory sizes, doing nothing.");
    return false;
  }

  // NB: positions are assumed to be in `victor_root/val_root` frame.
  ROS_INFO_STREAM("Executing dual gripper trajectory of length " << req.gripper1_points.size());
  for (size_t idx = 0; idx < req.gripper1_points.size(); ++idx)
  {
    auto ps = scene_->clonePlanningScene();
    auto const target = PointSequence{ ConvertTo<Eigen::Vector3d>(req.gripper1_points[idx]),
                                       ConvertTo<Eigen::Vector3d>(req.gripper2_points[idx]) };
    auto const traj = planner_->moveInWorldFrame(ps, target);
    followJointTrajectory(traj);
    res.merged_trajectory_empty = (traj.points.size() < 2);
  }

  ROS_INFO("Done trajectory");
  return true;
}

void DualGripperShim::followJointTrajectory(trajectory_msgs::JointTrajectory const& traj)
{
  std_msgs::String executing_action_str;
  executing_action_str.data = "Moving";
  talker_.publish(executing_action_str);
  if (traj.points.size() == 0)
  {
    ROS_INFO("Asked to follow trajectory of length 0; ignoring.");
    return;
  }
  if (!trajectory_client_->waitForServer(ros::Duration(3.0)))
  {
    ROS_WARN("Trajectory server not connected.");
  }

  control_msgs::FollowJointTrajectoryGoal goal;
  goal.trajectory = traj;
  for (const auto& name : goal.trajectory.joint_names)
  {
    // TODO: set thresholds for this task
    // NB: Dale: I think these are ignored by downstream code
    control_msgs::JointTolerance tol;
    tol.name = name;
    tol.position = 0.05;
    tol.velocity = 0.5;
    tol.acceleration = 1.0;
    goal.goal_tolerance.push_back(tol);
  }
  goal.goal_time_tolerance = traj_goal_time_tolerance_;
  ROS_INFO("Sending goal ...");
  trajectory_client_->sendGoalAndWait(goal);

  ROS_INFO("Goal finished");
}
