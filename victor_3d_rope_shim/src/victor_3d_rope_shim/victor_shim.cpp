#include "victor_3d_rope_shim/victor_shim.hpp"

#include <memory>
#include <vector>

namespace pm = peter_msgs;

std::pair<Eigen::Translation3d, Eigen::Translation3d> toGripperPositions(geometry_msgs::Point const& g1,
                                                                         geometry_msgs::Point const& g2)
{
  return { Eigen::Translation3d(g1.x, g1.y, g1.z), Eigen::Translation3d(g2.x, g2.y, g2.z) };
}

VictorShim::VictorShim(ros::NodeHandle nh, ros::NodeHandle ph)
{
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  victor_ = std::make_shared<VictorInterface>(nh, ph, tf_buffer_);

  // DualGripper control/exection
  {
    execute_traj_srv_ = nh.advertiseService("execute_dual_gripper_action", &VictorShim::executeTrajectory, this);
  }
}

////////////////////////////////////////////////////////////////////////////////

bool VictorShim::executeTrajectory(pm::DualGripperTrajectory::Request& req, pm::DualGripperTrajectory::Response& res)
{
  victor_->UpdatePlanningScene();

  if (req.gripper1_points.size() != req.gripper2_points.size())
  {
    ROS_WARN("Mismatched gripper trajectory sizes, doing nothing.");
    return false;
  }

  // NB: positions are assumed to be in `victor_root` frame.
  ROS_INFO_STREAM("Executing dual gripper trajectory of length " << req.gripper1_points.size());
  for (size_t idx = 0; idx < req.gripper1_points.size(); ++idx)
  {
    res.merged_trajectory_empty =
        victor_->moveInRobotFrame(toGripperPositions(req.gripper1_points[idx], req.gripper2_points[idx]));
  }

  ROS_INFO("Done trajectory");
  return true;
}
