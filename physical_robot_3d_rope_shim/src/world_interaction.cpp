#include <ros/ros.h>

#include "physical_robot_3d_rope_shim/dual_gripper_shim.hpp"
#include "physical_robot_3d_rope_shim/val_interface.hpp"

int main(int argc, char* argv[])
{
  // Read in all ROS parameters
  ros::init(argc, argv, "world_interaction");

  ros::NodeHandle nh;
  ros::NodeHandle ph("~");
  ros::AsyncSpinner spinner(1);
  spinner.start();

  auto shim = DualGripperShim(nh, ph);
  // shim.test();
  shim.gotoHome();
  shim.scene_->updatePlanningScene();
  shim.enableServices();

  ros::Rate rate(1);
  while (ros::ok())
  {
    shim.scene_->updatePlanningScene();
    rate.sleep();
  }

  return EXIT_SUCCESS;
}
