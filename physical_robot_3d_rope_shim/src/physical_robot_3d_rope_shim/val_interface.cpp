#include "physical_robot_3d_rope_shim/scene.hpp"
#include "physical_robot_3d_rope_shim/val_interface.hpp"

#include <arc_utilities/ros_helpers.hpp>

ValInterface::ValInterface(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<tf2_ros::Buffer> tf_buffer,
                           std::string const& group)
  : PlanningInterface(nh, ph, tf_buffer, group)
{
}

Eigen::VectorXd ValInterface::lookupQHome()
{
  if (planning_group_ == "left_arm")
  {
    auto left_home = ROSHelpers::GetVector<double>(nh_, "left_arm_home", std::vector<double>(7, 0));
    return Eigen::Map<Eigen::VectorXd>(left_home.data(), left_home.size());
  }
  else if (planning_group_ == "right_arm")
  {
    auto right_home = ROSHelpers::GetVector<double>(nh_, "right_arm_home", std::vector<double>(7, 0));
    return Eigen::Map<Eigen::VectorXd>(right_home.data(), right_home.size());
  }
  else if (planning_group_ == "torso")
  {
    auto torso_home = ROSHelpers::GetVector<double>(nh_, "torso_home", std::vector<double>(2, 0));
    return Eigen::Map<Eigen::VectorXd>(torso_home.data(), torso_home.size());
  }
  else if (planning_group_ == "both_arms")
  {
    auto torso_home = ROSHelpers::GetVector<double>(nh_, "torso_home", std::vector<double>(2, 0));
    auto left_home = ROSHelpers::GetVector<double>(nh_, "left_arm_home", std::vector<double>(7, 0));
    auto right_home = ROSHelpers::GetVector<double>(nh_, "right_arm_home", std::vector<double>(7, 0));
    Eigen::VectorXd home(16);
    home << Eigen::Map<Eigen::VectorXd>(torso_home.data(), torso_home.size()),
        Eigen::Map<Eigen::VectorXd>(left_home.data(), left_home.size()),
        Eigen::Map<Eigen::VectorXd>(right_home.data(), right_home.size());
    return home;
  }
  // add a new home group here for one arm and torso
  else
  {
    ROS_FATAL("Unknown planning group %s", planning_group_);
    return Eigen::VectorXd();
  }
}

void ValInterface::updateAllowedCollisionMatrix(collision_detection::AllowedCollisionMatrix& acm)
{
  (void)acm;
}
