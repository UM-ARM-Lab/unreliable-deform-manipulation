#include "physical_robot_3d_rope_shim/dual_gripper_shim.hpp"
#include "physical_robot_3d_rope_shim/val_interface.hpp"

#include <arc_utilities/ros_helpers.hpp>

#include "eigen_ros_conversions.hpp"
#include "assert.hpp"

namespace pm = peter_msgs;

DualGripperShim::DualGripperShim(ros::NodeHandle nh, ros::NodeHandle ph)
  : nh_(nh)
  , ph_(ph)
  , tf_buffer_(std::make_shared<tf2_ros::Buffer>())
  , tf_listener_(std::make_shared<tf2_ros::TransformListener>(*tf_buffer_))
{
  auto const robot_name = ROSHelpers::GetParam<std::string>(nh, "robot_name", "val");
  assert(robot_name == "val" || robot_name == "victor");

  if (robot_name == "victor")
  {
    // robot_ = std::make_shared<ValInterface>(nh, ph, tf_buffer_, "both_arms");
  }
  else if (robot_name == "val")
  {
    robot_ = std::make_shared<ValInterface>(nh_, ph_, tf_buffer_, "both_arms");
  }
  robot_->configureHomeState();
  scene_ = std::make_shared<Scene>(nh_, ph_, robot_);
}

////////////////////////////////////////////////////////////////////////////////

void DualGripperShim::test()
{
  auto const tool_transforms = robot_->getToolTransforms(robot_->home_state_);
  PointSequence target_positions(tool_transforms.size());
  for (auto idx = 0ul; idx < target_positions.size(); ++idx)
  {
    target_positions[idx] = tool_transforms[idx].translation() + 0.2 * Eigen::Vector3d::Random();
  }

  auto ps = scene_->clonePlanningScene();
  ps->setCurrentState(robot_->home_state_);
  auto const traj = robot_->moveInWorldFrame(ps, target_positions);
}

void DualGripperShim::enableServices()
{
  execute_traj_srv_ = nh_.advertiseService("execute_dual_gripper_action", &DualGripperShim::executeTrajectory, this);
  ROS_INFO("Ready for commands");
}

bool DualGripperShim::executeTrajectory(pm::DualGripperTrajectory::Request& req,
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
    auto const traj = robot_->moveInRobotFrame(ps, target);

    MPS_ASSERT(false && "Still needs to move");

    res.merged_trajectory_empty = (traj.points.size() == 0);
  }

  ROS_INFO("Done trajectory");
  return true;
}
