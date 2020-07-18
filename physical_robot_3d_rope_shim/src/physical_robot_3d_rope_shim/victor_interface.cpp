#include "physical_robot_3d_rope_shim/scene.hpp"
#include "physical_robot_3d_rope_shim/victor_interface.hpp"

#include <arc_utilities/ros_helpers.hpp>

VictorInterface::VictorInterface(ros::NodeHandle nh, ros::NodeHandle ph, std::shared_ptr<tf2_ros::Buffer> tf_buffer,
                                 std::string const& group)
  : PlanningInterface(nh, ph, tf_buffer, group)
{
}

Eigen::VectorXd VictorInterface::lookupQHome()
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
  else if (planning_group_ == "both_arms")
  {
    auto left_home = ROSHelpers::GetVector<double>(nh_, "left_arm_home", std::vector<double>(7, 0));
    auto right_home = ROSHelpers::GetVector<double>(nh_, "right_arm_home", std::vector<double>(7, 0));
    Eigen::VectorXd home(14);
    home << Eigen::Map<Eigen::VectorXd>(left_home.data(), left_home.size()),
        Eigen::Map<Eigen::VectorXd>(right_home.data(), right_home.size());
    return home;
  }
  else
  {
    ROS_FATAL("Unknown planning group %s", planning_group_);
    return Eigen::VectorXd();
  }
}

void VictorInterface::updateAllowedCollisionMatrix(collision_detection::AllowedCollisionMatrix& acm)
{
  std::vector<std::string> const VICTOR_TORSO_BODIES = {
    "victor_pedestal",
    "victor_base_plate",
    "victor_left_arm_mount",
    "victor_right_arm_mount",
  };

  // Disable collisions between the static obstacles and Victor's non-moving parts
  acm.setEntry(Scene::OBSTACLES_NAME, false);
  for (auto const& name : VICTOR_TORSO_BODIES)
  {
    acm.setEntry(name, Scene::OBSTACLES_NAME, true);
  }
}